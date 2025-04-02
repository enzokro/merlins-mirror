"""Merlin looks into the mirror..."""
from fasthtml.common import *
from monsterui.all import *
import os
import sys
import asyncio
import signal
import multiprocessing as mp
from dotenv import load_dotenv

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
# setup the app headers
hdrs = [theme, favicon_headers, full_screen_style, sse_script]

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
        # Main Image Container - takes all available space
        Div(
            Img(src="/static/logo.png", cls="image-fit"),
            id="image-container",
            cls="image-wrapper",
            hx_ext="sse",
            sse_connect="/generate",
            hx_swap="innerHTML",
            sse_swap="message",
        ),
        
        # Controls Section - fixed height at bottom
        DivFullySpaced(
            DivHStacked(
                # Left side form - HTMX adjusted for silent submission
                Div(
                    DivFullySpaced(
                        FormLabel(app_name, fr="prompt", cls="text-center justify-center text-2xl text-primary font-bold"),
                        cls="rounded-lg bg-secondary shadow-lg border",
                    ),
                    Input(id="prompt", name="prompt", placeholder="What do you see?", 
                            cls="w-full p-2 border rounded text-lg"),
                    Button("Generate", id="generate", type="button", cls=ButtonT.primary + ' border text-xl rounded-lg shadow-lg',
                            hx_post="/set_prompt",
                            hx_swap="none", 
                            hx_include="#prompt"),
                    cls="flex items-end space-x-2 flex-grow justify-center align-center",
                ),
                
                # Right side button - HTMX adjusted for silent submission
                Button("Refresh Latents", id="refresh", cls=ButtonT.secondary + ' border rounded-lg shadow-lg ml-4 mt-5',
                        hx_post="/refresh_latents",
                        hx_swap="none"),
                
                cls="w-full",
            ),
            cls="controls-wrapper",
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
            # Create image tag with base64 data
            data_uri = f"data:image/jpeg;base64,{result['data']}"
            img = Img(src=data_uri, cls="image-fit")
            yield sse_message(img)
            
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