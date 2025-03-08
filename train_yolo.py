#!/usr/bin/env python3
# train_yolo.py - YOLOv8 training script with NNI integration for hyperparameter tuning
import os
import sys
import json
import argparse
import yaml
import nni
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_parameters():
    """
    Get parameters from NNI and combine with any command-line arguments
    """
    # Get NNI parameters
    try:
        params = nni.get_next_parameter()
        logger.info(f"Received parameters from NNI: {params}")
    except Exception as e:
        logger.warning(f"Error getting parameters from NNI: {e}")
        params = {}

    return params


def validate_dataset(data_yaml):
    """
    Validate that dataset exists and is properly formatted
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config file not found: {data_yaml}")

    with open(data_yaml, 'r') as f:
        try:
            data_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in dataset config: {e}")

    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data_config:
            raise KeyError(f"Dataset config missing required key: {key}")

    # Check if dataset path exists
    dataset_path = data_config['path']
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    logger.info(f"Dataset validated successfully: {data_yaml}")
    logger.info(f"Classes: {data_config['names']}")

    # Check class balance
    analyze_dataset_balance(data_yaml, data_config)

    return data_config


def analyze_dataset_balance(data_yaml, data_config):
    """
    Analyze class distribution in the dataset
    """
    try:
        # Build the path to the labels directory
        base_path = data_config['path']
        train_labels = os.path.join(base_path, data_config['train'].replace('images', 'labels'))

        if not os.path.exists(train_labels):
            logger.warning(f"Labels path not found: {train_labels}")
            return

        # Count instances per class
        class_counts = {i: 0 for i in range(data_config['nc'])}
        total_instances = 0

        for label_file in os.listdir(train_labels):
            if not label_file.endswith('.txt'):
                continue

            with open(os.path.join(train_labels, label_file), 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            class_id = int(line.strip().split()[0])
                            class_counts[class_id] += 1
                            total_instances += 1
                        except (IndexError, ValueError) as e:
                            logger.warning(f"Invalid label format in {label_file}: {line.strip()}")

        # Log class distribution
        logger.info(f"Dataset contains {total_instances} labeled instances")
        for class_id, count in class_counts.items():
            percentage = 0 if total_instances == 0 else (count / total_instances) * 100
            logger.info(f"  Class {class_id} ({data_config['names'][class_id]}): {count} instances ({percentage:.2f}%)")

        # Check for significant imbalance
        avg_instances = total_instances / len(class_counts) if len(class_counts) > 0 else 0
        imbalanced_classes = []

        for class_id, count in class_counts.items():
            # Flag classes with less than 20% of the average instances
            if count < (avg_instances * 0.2) and avg_instances > 0:
                imbalanced_classes.append(class_id)

        if imbalanced_classes:
            class_names = [data_config['names'][i] for i in imbalanced_classes]
            logger.warning(f"Imbalanced classes detected: {class_names}")
            logger.warning("Consider using class weights or augmentation for these classes")

    except Exception as e:
        logger.warning(f"Error analyzing dataset balance: {e}")


def train_yolo(data_yaml, params):
    """
    Train a YOLOv8 model with the given parameters
    """
    # Validate dataset before training
    data_config = validate_dataset(data_yaml)

    # Default parameters
    model_size = params.get('model_size', 'yolov8n.pt')
    batch_size = params.get('batch_size', 16)
    epochs = params.get('epochs', 100)
    imgsz = params.get('imgsz', 640)

    # Learning rate parameters
    lr0 = params.get('lr0', 0.01)
    lrf = params.get('lrf', 0.01)

    # Regularization
    weight_decay = params.get('weight_decay', 0.0005)

    # Momentum parameters
    momentum = params.get('momentum', 0.937)

    # Warmup parameters
    warmup_epochs = params.get('warmup_epochs', 3.0)
    warmup_momentum = params.get('warmup_momentum', 0.8)
    warmup_bias_lr = params.get('warmup_bias_lr', 0.1)

    # Early stopping
    patience = params.get('patience', 50)

    # Data augmentation
    augment = params.get('augment', True)
    mixup = params.get('mixup', 0.0)
    mosaic = params.get('mosaic', 1.0)
    degrees = params.get('degrees', 0.0)
    translate = params.get('translate', 0.1)
    scale = params.get('scale', 0.5)
    shear = params.get('shear', 0.0)
    perspective = params.get('perspective', 0.0)
    flipud = params.get('flipud', 0.0)
    fliplr = params.get('fliplr', 0.5)

    logger.info(f"Training YOLOv8 model: {model_size}")
    logger.info(f"Hyperparameters: batch={batch_size}, epochs={epochs}, lr0={lr0}, imgsz={imgsz}")

    # Load the model
    try:
        model = YOLO(model_size)
        logger.info(f"Model loaded successfully: {model_size}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Train the model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            patience=patience,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            augment=augment,
            mixup=mixup,
            mosaic=mosaic,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            save=True,
            project='runs',
            name='train'
        )

        logger.info("Training completed successfully")

        # Log final metrics
        metrics = model.metrics
        logger.info(f"Final mAP50-95: {metrics.box.map}")
        logger.info(f"Final mAP50: {metrics.box.map50}")
        logger.info(f"Final precision: {metrics.box.precision}")
        logger.info(f"Final recall: {metrics.box.recall}")

        # Report final result to NNI
        # We maximize mAP (0-1 range, higher is better)
        nni.report_final_result(float(metrics.box.map))

        return metrics

    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Report a very low score to NNI in case of failure
        nni.report_final_result(0.0)
        raise


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 training with NNI hyperparameter tuning")
    parser.add_argument('--data_yaml', type=str, default='data/dataset.yaml',
                        help='Path to data.yaml file')
    args = parser.parse_args()

    # Get parameters from NNI
    params = get_parameters()

    # Train the model
    try:
        metrics = train_yolo(args.data_yaml, params)
        logger.info("Training and evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()