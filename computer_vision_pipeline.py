import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import logging
from typing import Tuple, Dict, List
import json
import torch
from torchvision import models
import torch.nn as nn

class StitchDetectionPipeline:
    def __init__(self, 
                 stitch_model_path: str = 'models/stitch_detection.pt',
                 defect_model_path: str = 'models/new_defect_detection_model.pt'):
        """
        Initialize the stitch detection pipeline
        """
        # Initialize YOLO models for stitch detection
        self.stitch_detector = YOLO(stitch_model_path)
        
        # Define the model architecture (same as during training)
        model = models.mobilenet_v2(weights=None)  # Do not download weights here, we'll load the custom weights
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # Update classifier
        
        # Move the model to the GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Initialize new defect detection model (assuming it's a PyTorch model now)
        self.defect_detector = model.load_state_dict(torch.load(defect_model_path))
        self.defect_detector.eval()  # Set model to evaluation mode
        
        # Initialize logging
        logging.basicConfig(filename='stitch_detection.log', level=logging.INFO)
        
        # Parameters for stitch analysis
        self.min_stitch_length = 2.0  # mm
        self.max_stitch_length = 4.0  # mm
        self.pixel_to_mm_ratio = 0.1  # will be calibrated
        
    def calibrate_camera(self, calibration_image: np.ndarray, known_distance_mm: float) -> float:
        """Calibrate the pixel to mm ratio using a reference image"""
        gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        points = np.int0(points)
        
        if len(points) >= 2:
            pixel_distance = np.linalg.norm(points[0] - points[1])
            self.pixel_to_mm_ratio = known_distance_mm / pixel_distance
            logging.info(f"Camera calibrated: {self.pixel_to_mm_ratio} mm/pixel")
            return self.pixel_to_mm_ratio
        else:
            logging.error("Calibration failed: Could not detect reference points")
            raise ValueError("Calibration failed")

    def detect_stitches(self, frame: np.ndarray) -> Dict:
        """Detect stitches in the frame and analyze their properties"""
        results = self.stitch_detector(frame)
        
        stitches = []
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            length_mm = ((x2 - x1) * self.pixel_to_mm_ratio)
            
            stitch = {
                'coordinates': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'length_mm': float(length_mm),
                'is_valid': self.min_stitch_length <= length_mm <= self.max_stitch_length
            }
            stitches.append(stitch)
        
        return {
            'stitches': stitches,
            'count': len(stitches),
            'average_length': np.mean([s['length_mm'] for s in stitches]) if stitches else 0,
            'timestamp': datetime.now().isoformat()
        }

    def detect_defects(self, frame: np.ndarray) -> Dict:
        """Detect fabric defects using the new defect detection model"""
        # Preprocess frame for the defect model (example, assuming model takes RGB image)
        transform = transforms.Compose([transforms.ToTensor()])
        frame_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = self.defect_detector(frame_tensor)
        
        # Assume output is a list of bounding boxes and class scores
        defects = []
        for detection in output[0]['boxes']:
            x1, y1, x2, y2 = detection[:4]
            conf = detection[4]
            cls = detection[5]
            defect_type = self.defect_detector.names[int(cls)]
            severity = 'high' if conf > 0.8 else 'medium' if conf > 0.6 else 'low'
            
            defects.append({
                'coordinates': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'type': defect_type,
                'severity': severity
            })
        
        return {
            'defects': defects,
            'count': len(defects),
            'timestamp': datetime.now().isoformat()
        }

    def analyze_alignment(self, frame: np.ndarray, reference_line: Tuple[float, float, float, float]) -> Dict:
        """Analyze stitch alignment against a reference line"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return {'alignment_score': 0, 'deviation': 0}
        
        ref_angle = np.arctan2(reference_line[3] - reference_line[1],
                             reference_line[2] - reference_line[0])
        
        deviations = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            deviation = abs(angle - ref_angle)
            deviations.append(min(deviation, np.pi - deviation))
        
        avg_deviation = np.mean(deviations)
        alignment_score = 100 * (1 - avg_deviation / (np.pi/2))
        
        return {
            'alignment_score': float(alignment_score),
            'deviation': float(avg_deviation),
            'timestamp': datetime.now().isoformat()
        }

    def process_frame(self, frame: np.ndarray, reference_line: Tuple[float, float, float, float]) -> Dict:
        """Process a single frame and return all analyses"""
        stitch_data = self.detect_stitches(frame)
        defect_data = self.detect_defects(frame)
        alignment_data = self.analyze_alignment(frame, reference_line)
        
        return {
            'stitch_analysis': stitch_data,
            'defect_analysis': defect_data,
            'alignment_analysis': alignment_data,
            'timestamp': datetime.now().isoformat()
        }

    def visualize_results(self, frame: np.ndarray, analysis_results: Dict) -> np.ndarray:
        """Visualize detection results on the frame"""
        output = frame.copy()
        
        # Draw stitches
        for stitch in analysis_results['stitch_analysis']['stitches']:
            x1, y1, x2, y2 = stitch['coordinates']
            color = (0, 255, 0) if stitch['is_valid'] else (0, 0, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw defects
        for defect in analysis_results['defect_analysis']['defects']:
            x1, y1, x2, y2 = defect['coordinates']
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output, f"{defect['type']}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw alignment score
        score = analysis_results['alignment_analysis']['alignment_score']
        cv2.putText(output, f"Alignment: {score:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return output
