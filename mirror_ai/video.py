"""Captures frames from webcam in a separate thread"""
import time
import threading
from PIL import Image
import cv2


class VideoStreamer:
    """Captures frames from webcam in a separate thread"""
    def __init__(self, camera_id=0, width=1920, height=1080, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.cap = None
        
        # Start the capture thread
        self.start()
        
    def start(self):
        """Starts the video capture thread"""
        if self.thread is not None and self.thread.is_alive():
            return  # already running
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _capture_loop(self):
        """Thread function that continuously captures frames"""
        self.cap = cv2.VideoCapture(self.camera_id)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite('/app/persist/camera_00_raw.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                # Convert to RGB (from BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite('/app/persist/camera_01_rgb.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                # Convert to PIL Image
                pil_image = Image.fromarray(frame)
                pil_image.save('/app/persist/camera_02_pil.jpg')
                
                # Update the current frame
                with self.lock:
                    self.frame = pil_image
            
            # Small sleep to reduce CPU usage
            time.sleep(0.01)
    
    def get_current_frame(self):
        """Gets the current frame as a PIL Image"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        """Explicitly stops capture thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None