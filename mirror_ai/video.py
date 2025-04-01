from .config import VideoConfig
import cv2 
from PIL import Image
import threading
from . import config 
from .utils import interpolate_images

class VideoStreamer:
    """
    Class for capturing video frames from a camera or video file.
    
    This class provides methods to initialize a video capture device and
    retrieve frames from it as PIL images.
    """
    
    def __init__(self, device_path=VideoConfig.VIDEO_PATH):
        """
        Initialize the video streamer.
        
        Args:
            device_path: Path to the video capture device or index for webcam
        """
        self.cap = cv2.VideoCapture(device_path)
        if not self.cap.isOpened():
            raise ValueError(f"Video source cannot be opened: {device_path}")
            
        # Set capture properties if defined in config
        if hasattr(VideoConfig, 'CAP_PROPS') and isinstance(VideoConfig.CAP_PROPS, dict):
            for prop, value in VideoConfig.CAP_PROPS.items():
                try:
                    prop_id = getattr(cv2, prop)
                    self.cap.set(prop_id, value)
                except AttributeError:
                    print(f"Unknown OpenCV property: {prop}")
        

    def get_current_frame(self):
        """
        Get the current frame from the video source.
        
        Returns:
            PIL Image or None if frame cannot be read
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert from BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create PIL image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize if needed
        if (pil_image.width != VideoConfig.WIDTH or pil_image.height != VideoConfig.HEIGHT) and hasattr(VideoConfig, 'WIDTH') and hasattr(VideoConfig, 'HEIGHT'):
            pil_image = pil_image.resize((VideoConfig.WIDTH, VideoConfig.HEIGHT))
            
        return pil_image

    def release(self):
        """Release the video capture resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            print("Video capture released")


# Video processing thread
class VideoProcessingThread(threading.Thread):
    def __init__(self, shared_resources, device_path=VideoConfig.VIDEO_PATH):
        super().__init__(daemon=True)
        self.shared_resources = shared_resources
        self.stop_event = shared_resources.stop_event
        self.video_streamer = VideoStreamer(device_path)
        self.previous_frame = None

    def run(self):
        while not self.stop_event.is_set():
            current_frame = self.video_streamer.get_current_frame()
            if current_frame is None:
                continue
            
            current_prompt = self.shared_resources.get_prompt()

            if current_prompt:
                transformed_frame = self.shared_resources.image_transformer.transform(
                    current_frame,
                    current_prompt,
                )
            else:
                transformed_frame = current_frame

            if self.previous_frame is not None:
                final_frame = interpolate_images(self.previous_frame, transformed_frame, config.FRAME_BLEND)
            else:
                final_frame = transformed_frame

            self.shared_resources.update_frame(final_frame)
            self.previous_frame = final_frame

        self.video_streamer.release()

