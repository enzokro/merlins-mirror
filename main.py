"""Main application for Merlin's Mirror"""
from fasthtml.common import *
from monsterui.all import *
import os
import io
import sys
import asyncio
import signal
import multiprocessing as mp
from dotenv import load_dotenv
from PIL import Image


from merlin import go_merlin
from mirror_ai import config

# load and set the environment variables
load_dotenv()
app_name = os.getenv("APP_NAME", "Merlin's Mirror")
theme_name = os.getenv("THEME", "violet").lower()

# queue to gather results
async_queue = asyncio.Queue()
# communication queues
request_queue = mp.Queue()
result_queue = mp.Queue()

# shutdown event
shutdown_event = signal_shutdown()

# set the MonsterUI theme
theme = getattr(Theme, theme_name).headers(mode="light")
# set the favicon
favicon_headers = Favicon(light_icon="/static/logo.png", dark_icon="/static/logo.png")
# makes our app full-screen
full_screen_style = Link(rel="stylesheet", href="/static/styles.css")
# sse to emit transformed images
sse_script = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")

canvas_script = Script("""
    // Canvas rendering and SSE handling
    document.addEventListener('DOMContentLoaded', function() {
        // Get canvas elements
        const mainCanvas = document.getElementById('main-canvas');
        const bgCanvas = document.getElementById('bg-canvas');
        const initialLogo = document.getElementById('initial-logo');
        
        // Get rendering contexts
        const mainCtx = mainCanvas.getContext('2d');
        const bgCtx = bgCanvas.getContext('2d');
        
        // Set initial dimensions based on container
        function updateCanvasDimensions() {
            const container = document.getElementById('canvas-container');
            if (container) {
                // Set dimensions based on container size for responsive behavior
                const width = container.clientWidth;
                const height = container.clientHeight;
                
                mainCanvas.width = width;
                mainCanvas.height = height;
                bgCanvas.width = width;
                bgCanvas.height = height;
            }
        }
        
        // Initial size update
        updateCanvasDimensions();
        
        // Update on window resize for responsiveness
        window.addEventListener('resize', updateCanvasDimensions);
        
        // Function to draw image with proper aspect ratio
        function drawImageProperly(img, canvas, ctx, expand = 0) {
            const imgRatio = img.width / img.height;
            const canvasRatio = canvas.width / canvas.height;
            
            let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
            
            // Determine dimensions to maintain aspect ratio
            if (imgRatio > canvasRatio) {
                // Image is wider than canvas (relative to height)
                drawHeight = canvas.height + expand * 2;
                drawWidth = drawHeight * imgRatio;
                offsetX = (canvas.width - drawWidth) / 2;
                offsetY = -expand;
            } else {
                // Image is taller than canvas (relative to width)
                drawWidth = canvas.width + expand * 2;
                drawHeight = drawWidth / imgRatio;
                offsetX = -expand;
                offsetY = (canvas.height - drawHeight) / 2;
            }
            
            // Draw the image centered and scaled properly
            ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
        }
        
        // Listen for SSE events containing image data
        document.body.addEventListener('sse:message', function(event) {
            // Get base64 image data from the event
            const imageData = event.detail.data;
            
            // Hide the initial logo once we start receiving images
            if (initialLogo && initialLogo.style.display !== 'none') {
                initialLogo.style.display = 'none';
            }
            
            // Create image object to load the data
            const img = new Image();
            
            img.onload = function() {
                // Use requestAnimationFrame for smooth rendering
                requestAnimationFrame(() => {
                    // Clear previous canvas content
                    mainCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
                    bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
                    
                    // Draw main image
                    drawImageProperly(img, mainCanvas, mainCtx);
                    
                    // Draw background with blur effect
                    bgCtx.filter = 'blur(10px)';
                    // Draw slightly larger to avoid blur edges
                    drawImageProperly(img, bgCanvas, bgCtx, 10);
                    bgCtx.filter = 'none';
                });
            };
            
            img.onerror = function(err) {
                console.error('Error loading image:', err);
            };
            
            // Set image source from base64 data
            img.src = 'data:image/jpeg;base64,' + imageData;
        });
    });
""")

# setup the app headers
hdrs = [theme, favicon_headers, full_screen_style, sse_script, canvas_script]

# create the merlin process
merlin = mp.Process(
    target=go_merlin,
    args=(request_queue, result_queue),
    daemon=True
)

# captures merlin's reflections
async def poll_result_queue(result_queue, async_queue, shutdown_event):
    """Bridge between multiprocessing queue and asyncio queue"""
    while not shutdown_event.is_set():
        # check for new results
        try:
            if not result_queue.empty():
                res = result_queue.get_nowait()
                await async_queue.put(res)
        except Exception as e:
            print(f"Error polling result queue: {e}")
        
        # small sleep to prevent CPU spinning
        await asyncio.sleep(0.01)

async def startup():
    "Starts the merlin process and handles incoming results."
    merlin.start()
    asyncio.create_task(poll_result_queue(result_queue, async_queue, shutdown_event))

# gracefully shuts down the app
async def shutdown():
    "Merlin can rest."
    shutdown_event.set()
    request_queue.put({"type": config.REQUEST_SHUTDOWN})

# create the mirror app
app, rt = fast_app(
    hdrs=hdrs,
    on_startup=startup,
    on_shutdown=shutdown,
)

# main page
@rt('/')
def index():
    return Title(app_name), Div(

        # Main Canvas Container
        Div(
            # Background canvas for blurred effect
            Canvas(
                id="bg-canvas", 
                cls="canvas-background"
            ),
            # Foreground canvas for main display
            Canvas(
                id="main-canvas", 
                cls="canvas-main"
            ),
            # Initial logo display that will be hidden when canvases are active
            Img(
                src="/static/logo.png", 
                id="initial-logo",
                cls="initial-image"
            ),
            id="canvas-container",
            cls="canvas-wrapper",
            hx_ext="sse",
            sse_connect="/generate"
            # No more innerHTML swap or message swap - we'll handle it in JavaScript
        ),
        # Controls Section - fixed height at bottom
        Div(
            # Two-column layout
            Div(
                # Left side: Simple, direct form structure
                Form(
                    # Input and button in a row
                    Div(
                        Input(id="prompt", name="prompt", placeholder="What do you see?", 
                            cls="flex-grow p-2 border rounded text-lg"),
                        Button("Generate", id="generate", type="submit", 
                            cls=ButtonT.primary + ' border text-xl rounded-lg shadow-lg ml-2'),
                        cls="flex w-full"
                    ),
                    hx_post="/set_prompt",
                    hx_swap="none",
                    cls="flex-grow flex flex-col",
                ),
                
                # Right side button
                Button("Refresh Scene", id="refresh", 
                    cls=ButtonT.secondary + ' border rounded-lg shadow-lg ml-4',
                    hx_post="/refresh_latents",
                    hx_swap="none"),
                
                cls="flex items-end w-full"
            ),
            cls="controls-wrapper"
        ),
        
        cls="main-container",
    )

# sets the prompt
@rt('/set_prompt')
def set_prompt(prompt: str):
    print(f"Web: Setting prompt: {prompt}")
    request_queue.put({
        "type": config.REQUEST_SET_PROMPT,
        "prompt": prompt
    })
    return ""  # Empty response for HTMX

# refreshes the latents for a touch up
@rt('/refresh_latents')
def refresh_latents():
    print("Web: Refreshing latents")
    request_queue.put({
        "type": config.REQUEST_REFRESH_LATENTS
    })
    return ""  # Empty response for HTMX

# sends the generated images in an sse stream
async def generate():
    """Generates SSE events with transformed image results."""
    while not shutdown_event.is_set():
        result = await async_queue.get()
        
        if result["type"] == config.RESULT_FRAME:
            encoded_img = result['data']
            yield sse_message(encoded_img)
            
        elif result["type"] == config.RESULT_ERROR:
            # Display error message
            error_div = Div(
                P(f"Error: {result['message']}", cls="text-red-500"),
                cls="p-4 bg-red-100 rounded"
            )
            yield sse_message(error_div)
            
        elif result["type"] == config.RESULT_STATUS:
            # Status updates
            status_div = Div(
                P(f"Status: {result['status']}", cls="text-blue-500"),
                cls="p-4 bg-blue-100 rounded"
            )
            yield sse_message(status_div)

# SSE endpoint
@rt("/generate")
async def merlin_looks_into_the_mirror():
    return EventStream(generate())

# setup for graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down...")
    request_queue.put({"type": config.REQUEST_SHUTDOWN})
    merlin.join(timeout=5)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# what did Merlin see?
serve()