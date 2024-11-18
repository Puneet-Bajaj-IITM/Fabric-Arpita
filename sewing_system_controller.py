import cv2
import numpy as np
from threading import Thread, Lock
import queue
import time
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime
from barcode_data_system import BarcodeDataSystem
from computer_vision_pipeline import StitchDetectionPipeline

class SewingSystemController:
    def __init__(self):
        """Initialize the main sewing system controller"""
        # Initialize components
        self.stitch_detector = StitchDetectionPipeline()
        self.data_system = BarcodeDataSystem()
        
        # Initialize cameras
        self.top_camera = cv2.VideoCapture(0)  # Adjust indices as needed
        self.bottom_camera = cv2.VideoCapture(1)
        
        # Set camera properties
        self._set_camera_properties()
        
        # Threading setup
        self.frame_lock = Lock()
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.is_running = False
        
        # Session state
        self.current_session: Optional[str] = None
        self.reference_line: Optional[Tuple] = None
        self.light_settings = {'top': 80, 'bottom': 80}
        
        # Setup logging
        logging.basicConfig(
            filename='sewing_system.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _set_camera_properties(self):
        """Configure camera settings"""
        for camera in [self.top_camera, self.bottom_camera]:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Manual exposure
            camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust as needed
            
    def set_light_intensity(self, camera: str, value: int):
        """Adjust lighting intensity for specified camera"""
        self.light_settings[camera] = max(0, min(100, value))
        # Implement hardware control for lighting here
        logging.info(f"Set {camera} camera light intensity to {value}")
        
    def set_reference_line(self, points: Tuple[float, float, float, float]):
        """Set reference line for alignment checking"""
        self.reference_line = points
        logging.info(f"Set reference line to {points}")
        
    def start_session(self, barcode_id: Optional[str] = None):
        """Start a new sewing session"""
        if barcode_id is None:
            # Try to scan barcode from top camera
            ret, frame = self.top_camera.read()
            if ret:
                fabric_data = self.data_system.scan_barcode(frame)
                if fabric_data:
                    barcode_id = fabric_data['barcode_id']
        
        if barcode_id:
            self.current_session = self.data_system.start_session(barcode_id)
            self.is_running = True
            
            # Start processing threads
            self.frame_thread = Thread(target=self._frame_capture_loop)
            self.process_thread = Thread(target=self._processing_loop)
            
            self.frame_thread.start()
            self.process_thread.start()
            
            logging.info(f"Started session {self.current_session}")
            return self.current_session
        else:
            raise ValueError("No valid barcode detected")
            
    def stop_session(self):
        """Stop the current sewing session"""
        if self.current_session:
            self.is_running = False
            
            # Wait for threads to finish
            self.frame_thread.join()
            self.process_thread.join()
            
            # Clear queues
            while not self.frame_queue.empty():
                self.frame_queue.get()
            while not self.result_queue.empty():
                self.result_queue.get()
                
            # End session in database
            self.data_system.end_session(self.current_session)
            
            # Generate report
            report = self.data_system.generate_report(self.current_session)
            
            logging.info(f"Ended session {self.current_session}")
            self.current_session = None
            
            return report
            
    def _frame_capture_loop(self):
        """Continuous frame capture from both cameras"""
        while self.is_running:
            ret_top, frame_top = self.top_camera.read()
            ret_bottom, frame_bottom = self.bottom_camera.read()
            
            if ret_top and ret_bottom:
                timestamp = datetime.now()
                with self.frame_lock:
                    if not self.frame_queue.full():
                        self.frame_queue.put({
                            'top': frame_top,
                            'bottom': frame_bottom,
                            'timestamp': timestamp
                        })
            
            time.sleep(1/30)  # Limit to 30 FPS
            
    def _processing_loop(self):
        """Continuous processing of captured frames"""
        while self.is_running:
            try:
                frames = self.frame_queue.get(timeout=1)
                
                # Process top camera frame
                if self.reference_line:
                    analysis_results = self.stitch_detector.process_frame(
                        frames['top'],
                        self.reference_line
                    )
                    
                    # Log measurements
                    self.data_system.log_measurement(
                        self.current_session,
                        analysis_results
                    )
                    
                    # Add visualization
                    vis_frame = self.stitch_detector.visualize_results(
                        frames['top'],
                        analysis_results
                    )
                    
                    # Put results in queue for GUI
                    self.result_queue.put({
                        'frame': vis_frame,
                        'analysis': analysis_results,
                        'timestamp': frames['timestamp']
                    })
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                
    def get_latest_results(self) -> Optional[Dict]:
        """Get the latest processing results"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
            
    def calibrate_system(self):
        """Perform system calibration"""
        # Capture calibration frame
        ret, frame = self.top_camera.read()
        if ret:
            try:
                # Use known reference distance (e.g., calibration pattern)
                known_distance_mm = 10.0  # Example reference distance
                self.stitch_detector.calibrate_camera(frame, known_distance_mm)
                logging.info("System calibration completed successfully")
                return True
            except Exception as e:
                logging.error(f"Calibration failed: {str(e)}")
                return False
        return False
        
    def __del__(self):
        """Cleanup resources"""
        self.is_running = False
        if hasattr(self, 'frame_thread'):
            self.frame_thread.join()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            
        self.top_camera.release()
        self.bottom_camera.release()
