# ml_process.py
import time
from io import BytesIO
import base64
import numpy as np
from PIL import Image
from mirror_ai.utils import resize

def go_merlin(request_queue, result_queue):
    """Processes webcame frames into transformed images."""
    try:
        # Import here to avoid loading these in the web process
        from mirror_ai.pipeline import ImagePipeline
        from mirror_ai.video import VideoStreamer
        from mirror_ai.utils import convert_to_pil_image
        from mirror_ai import config
        
        print("Initializing ML pipeline...")
        
        # Initialize components
        pipeline = ImagePipeline()
        pipeline.load()
        video_streamer = VideoStreamer()
        
        current_prompt = None
        running = True
        
        print("ML process ready")
        # Send status update to web process
        result_queue.put({
            "type": config.RESULT_STATUS,
            "status": "ready"
        })
        
        # Main processing loop
        while running:
            # check for new requests
            try:
                while not request_queue.empty():
                    request = request_queue.get_nowait()
                    if request["type"] == config.REQUEST_SHUTDOWN:
                        running = False
                        break
                    elif request["type"] == config.REQUEST_SET_PROMPT:
                        current_prompt = request["prompt"]
                        print(f"New prompt: {current_prompt}")
                    elif request["type"] == config.REQUEST_REFRESH_LATENTS:
                        pipeline.refresh_latents()
                        print("Latents refreshed")
            except Exception as e:
                print(f"Error processing request: {e}")
            
            # Skip processing if we don't have a prompt yet
            if not current_prompt or not running:
                time.sleep(0.1)
                continue
                
            # process camera frames
            try:
                camera_frame = video_streamer.get_current_frame()
                if camera_frame is not None:
                    # transform the webcam image
                    processed_frame = pipeline.generate(current_prompt, camera_frame)
                    
                    # convert to storable format
                    pil_frame = convert_to_pil_image(processed_frame)
                    # resize keeping aspect ratio
                    pil_frame = resize(pil_frame, width=config.DISPLAY_WIDTH, height=config.DISPLAY_HEIGHT)
                    
                    # turn into base64 (easier to send through queue)
                    img_byte_arr = BytesIO()
                    pil_frame.save(img_byte_arr, format='JPEG', quality=90)
                    img_byte_arr.seek(0)
                    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    
                    # send the image back
                    result_queue.put({
                        "type": config.RESULT_FRAME,
                        "data": encoded_img
                    })
            except Exception as e:
                print(f"Error during frame processing: {e}")
                result_queue.put({
                    "type": config.RESULT_ERROR,
                    "message": str(e)
                })
                time.sleep(0.5)  # Avoid rapid error loops
    
    except Exception as e:
        print(f"Fatal error in ML process: {e}")
        # Try to notify web process
        try:
            result_queue.put({
                "type": config.RESULT_ERROR,
                "message": f"ML process crashed: {str(e)}"
            })
        except:
            pass
    
    finally:
        print("ML process shutting down")
        # Clean up resources
        try:
            video_streamer.release()
        except:
            pass