"""Captures frames from webcam in a separate thread"""
import time
import threading
from PIL import Image
import cv2


class VideoStreamer:
    """Captures frames from webcam in a separate thread"""
    def __init__(self, camera_id=0, width=640, height=480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.cap = None
        
        # Start the capture thread
        self.start()
        
    def start(self):
        """Start the video capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _capture_loop(self):
        """Thread function that continuously captures frames"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert to RGB (from BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Update the current frame
                with self.lock:
                    self.frame = pil_image
            
            # Small sleep to reduce CPU usage
            time.sleep(0.01)
    
    def get_current_frame(self):
        """Get the current frame as a PIL Image"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def release(self):
        """Release resources"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()