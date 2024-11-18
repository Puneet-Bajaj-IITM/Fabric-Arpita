import os
import yaml
from ultralytics import YOLO
import torch
import albumentations as A
from pathlib import Path
import logging
import argparse
from datetime import datetime
import cv2
import numpy as np

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def create_dataset_config(data_path):
    """Create YAML configuration for the dataset"""
    dataset_config = {
        'path': data_path,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'Crochet-Stitch'  # Single class as per dataset description
        },
        'nc': 1  # number of classes
    }
    
    config_path = 'dataset_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f)
    
    return config_path

def get_preprocessing_pipeline():
    """Define preprocessing pipeline matching the dataset"""
    return A.Compose([
        A.Resize(640, 640, p=1.0),  # Resize to 640x640 with white edges
        A.ToGray(p=1.0),  # Convert to grayscale
        A.CLAHE(p=1.0),   # Adaptive contrast enhancement
    ])

def get_augmentation_pipeline():
    """Define augmentation pipeline matching the dataset exactly"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # 50% probability of horizontal flip
        A.OneOf([
            A.Rotate(limit=0, p=1.0),  # No rotation
            A.Rotate(limit=90, p=1.0),  # Clockwise 90 degrees
            A.Rotate(limit=-90, p=1.0)  # Counter-clockwise 90 degrees
        ], p=0.33),
        A.RandomResizedCrop(
            height=640,
            width=640,
            scale=(0.8, 1.0),  # Random crop between 0 and 20 percent
            p=1.0
        ),
        A.Rotate(limit=17, p=1.0),  # Random rotation between -17 and +17 degrees
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # ±20% brightness adjustment
            contrast_limit=0.0,
            p=1.0
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.0,
            contrast_limit=0.15,  # ±15% exposure adjustment
            p=1.0
        )
    ])

def train_yolo(args):
    """Main training function"""
    logging.info("Starting YOLO training process...")
    
    # Create dataset configuration
    config_path = create_dataset_config(args.data_path)
    logging.info(f"Created dataset configuration at {config_path}")
    
    # Initialize YOLO model with v8m architecture
    model = YOLO('yolov8m.pt')  # Using medium model as specified in dataset name
    logging.info("Loaded YOLOv8m model")
    
    # Configure training parameters
    train_args = {
        'data': config_path,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': 900,  # Using 900 image size as specified in dataset name
        'patience': 20,  # Early stopping patience
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': args.num_workers,
        'project': args.project_name,
        'name': f'crochet_stitch_detection_v37_{datetime.now().strftime("%Y%m%d")}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': args.learning_rate,
        'cos_lr': True,
        'cache': True,
        'mosaic': 1.0,  # Enable mosaic augmentation as specified in dataset name
        'degrees': 17.0,  # Max rotation degrees
        'scale': 0.2,    # Random scaling
        'flipud': 0.0,   # No vertical flip
        'fliplr': 0.5,   # 50% horizontal flip
        'mixup': 0.0,    # No mixup
        'copy_paste': 0.0, # No copy-paste
    }
    
    # Start training
    logging.info("Starting training with parameters:")
    logging.info(train_args)
    results = model.train(**train_args)
    
    # Save training results
    logging.info("Training completed. Saving results...")
    model.save(f'{args.project_name}/weights/best.pt')
    
    # Validate the model
    logging.info("Running validation...")
    val_results = model.val()
    
    return results, val_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO Crochet Stitch Detection Training')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker threads')
    parser.add_argument('--project_name', type=str, default='crochet_stitch_detection',
                      help='Project name for saving results')
    return parser.parse_args()

if __name__ == '__main__':
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Run training
        results, val_results = train_yolo(args)
        
        # Log final metrics
        logging.info("Training completed successfully")
        logging.info(f"Final mAP: {val_results.maps}")
        logging.info(f"Best model saved at: {args.project_name}/weights/stitch-detection.pt")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise