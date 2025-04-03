"""Merlin looks into the mirror..."""
import os
from dotenv import load_dotenv

load_dotenv()
IMAGE_DEBUG_PATH = os.getenv("IMAGE_DEBUG_PATH")

def go_merlin(request_queue, result_queue):
    """Turns webcam frames into AI-generated images."""
    try:
        # import what we need here to avoid heavy imports in the web process
        import traceback
        import time
        import base64
        from io import BytesIO
        import torch
        from mirror_ai.pipeline import ImagePipeline
        from mirror_ai.video import VideoStreamer
        from mirror_ai.utils import convert_to_pil_image, resize
        from mirror_ai import config
        
        print("Initializing Merlin...")
        
        # set up the image transformer and the webcam stream
        pipeline = ImagePipeline()
        pipeline.load(config.SCHEDULER_NAME)
        video_streamer = VideoStreamer(
            camera_id=config.CAMERA_ID,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS
        )
        
        # start with no prompt, and default running
        current_prompt = None
        running = True
        
        print("Merlin is ready to ponder.")
        result_queue.put({
            "type": config.RESULT_STATUS,
            "status": "ready"
        })
        
        # generate images in a loop
        while running:
            # process incoming requests from the web app
            try:
                while not request_queue.empty():
                    request = request_queue.get_nowait()

                    # shut down Merlin
                    if request["type"] == config.REQUEST_SHUTDOWN:
                        running = False
                        break

                    # change the prompt
                    elif request["type"] == config.REQUEST_SET_PROMPT:
                        current_prompt = request["prompt"]
                        print(f"New prompt: {current_prompt}")

                    # refresh the latents
                    elif request["type"] == config.REQUEST_REFRESH_LATENTS:
                        pipeline.refresh_latents()
                        print("Latents refreshed")

            except Exception as e:
                print(f"Error in Merlin process: {traceback.format_exc()}")
            
            # wait until we have a prompt
            if not current_prompt or not running:
                time.sleep(0.1)
                continue
                
            # process camera frames
            try:
                camera_frame = video_streamer.get_current_frame()
                camera_frame.save(f'{IMAGE_DEBUG_PATH}/camera_04_queue_shape_{camera_frame.size}.jpg')
                if camera_frame is not None:
                    # transform the webcam image
                    processed_frame = pipeline.generate(current_prompt, camera_frame)
                    # processed_frame.save(f'{IMAGE_DEBUG_PATH}/camera_05_generated_shape_{processed_frame.size}.jpg')
                    
                    # convert to storable format
                    # pil_frame = convert_to_pil_image(processed_frame)
                    pil_frame = processed_frame
                    # pil_frame.save(f'{IMAGE_DEBUG_PATH}/camera_06_pil_shape_{pil_frame.size}.jpg')

                    # resize keeping aspect ratio
                    pil_frame = pil_frame.resize((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                    # pil_frame.save(f'{IMAGE_DEBUG_PATH}/camera_07_resized_shape_{pil_frame.size}.jpg')

                    # turn into base64 (easier to send through queue)
                    img_byte_arr = BytesIO()
                    pil_frame.save(img_byte_arr, format='JPEG', quality=config.JPEG_QUALITY)
                    img_byte_arr.seek(0)
                    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    
                    # send the image back
                    result_queue.put({
                        "type": config.RESULT_FRAME,
                        "data": encoded_img
                    })

            except Exception as e:
                print(f"Error during frame processing: {traceback.format_exc()}")
                result_queue.put({
                    "type": config.RESULT_ERROR,
                    "message": str(e)
                })
                time.sleep(0.5)  # avoid rapid error loops
    
    except Exception as e:
        print(f"Fatal error in Merlin process: {traceback.format_exc()}")
        # notify the web process
        try:
            result_queue.put({
                "type": config.RESULT_ERROR,
                "message": f"Merlin crashed 😔: {traceback.format_exc()}"
            })
        except:
            pass
    
    finally:
        print("Merlin going to sleep...")
        try:
            # cleanup the webcam
            if video_streamer is not None:
                video_streamer.stop()

            # cleanup the pipeline
            if pipeline is not None:
                del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except:
            pass