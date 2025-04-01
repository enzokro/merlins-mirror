import asyncio
import base64
from io import BytesIO
from PIL import Image
from fasthtml.common import *
from monsterui.all import *
from dotenv import load_dotenv
from mirror_ai.pipeline import ImagePipeline
from mirror_ai.video import VideoStreamer
from mirror_ai.utils import SharedResources, convert_to_pil_image
from mirror_ai import config

# load the env vars
load_dotenv()

# setup the app
app_name = os.getenv("APP_NAME", "Merlin's Mirror")
theme_name = os.getenv("THEME", "violet").lower()

# setup the favicon
favicon_headers = Favicon(
    light_icon="/static/logo.png",
    dark_icon="/static/logo.png"
)

# setup the theme
theme = getattr(Theme, theme_name).headers(mode="light")

# add explicit css to control viewport and image behavior
full_screen_style = Link(rel="stylesheet", href="/static/styles.css")

# SSE to send processed frames to the client
sse_script = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")

hdrs = [
    theme,
    favicon_headers,
    full_screen_style,
    sse_script,
]

# create the app
app, rt = fast_app(
    hdrs=hdrs,
)

# create the pipeline
pipeline = None #ImagePipeline()
# pipeline.load()

# create the shared resources
shared_resources = SharedResources(pipeline)
# grabs frames from the webcam
video_streamer = VideoStreamer()

@rt('/')
def get():
    return Title(app_name), Div(
        
        # Main Image Container - takes all available space
        Div(
            Img(src="/static/logo.png", cls="image-fit"),
            id="image-container",
            cls="image-wrapper",
            # hx_ext="sse",
            # sse_connect="/generate",
            # hx_swap="innerHTML",
            # sse_swap="message",
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
                    # P(app_name, cls="text-xl font-bold text-purple-700 mb"),
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

@rt('/set_prompt')
def post(prompt: str):
    print(f"prompt: {prompt}")
    shared_resources.update_prompt(prompt)

@rt('/refresh_latents')
def post():
    print("Refreshing latents...")
    shared_resources.refresh_latents()

shutdown_event = signal_shutdown()
async def generate():
    while not shutdown_event.is_set():
        # # TODO: remove when working
        # yield sse_message(P("Generating..."))
        # await asyncio.sleep(1)
        # continue

        # get the current frame from the webcam, and latest prompt
        camera_frame = video_streamer.get_current_frame()
        prompt = shared_resources.get_prompt()

        # Ensure camera_frame is not None and prompt is available
        if camera_frame is None or prompt is None:
            await asyncio.sleep(0.1) # Avoid busy-waiting if resources aren't ready
            continue

        try:
            frame = shared_resources.generate_frame(prompt, camera_frame)

            # Convert the processed frame (assuming PIL or convertible) to base64 JPEG
            img_byte_arr = BytesIO()
            pil_frame = convert_to_pil_image(frame) # Convert if necessary
            # Resize using constants from config
            pil_frame = pil_frame.resize((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)) 
            pil_frame.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{encoded_img}"

            # Create the Img tag with the base64 data URI
            img = Img(src=data_uri, cls="image-fit")
            yield sse_message(img)

        except Exception as e:
            print(f"Error during frame generation or encoding: {e}")
            # Optionally yield an error message or placeholder image
            await asyncio.sleep(0.5) # Wait a bit before retrying on error

@rt("/generate")
async def get(): return EventStream(generate())


serve()