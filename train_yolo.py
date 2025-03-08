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

    # Make sure all parameters from search_space.json are present
    # If not provided by NNI, use default values
    defaults = {
        'model_size': 'yolov8n.pt',
        'batch_size': 16,
        'epochs': 100,
        'imgsz': 640,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'patience': 50,
        'augment': True,
        'mosaic': 1.0,
        'mixup': 0.0,
        'scale': 0.5,
        'fliplr': 0.5,
        'translate': 0.1,
        'degrees': 0.0,  # Added missing parameter
        'shear': 0.0,  # Added missing parameter
        'perspective': 0.0,  # Added missing parameter
        'flipud': 0.0  # Added for completeness
    }

    # Update defaults with provided parameters
    for key, default_value in defaults.items():
        if key not in params:
            params[key] = default_value
            logger.info(f"Using default value for {key}: {default_value}")

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


class NNIReportCallback:
    """
    Callback class to report intermediate results to NNI
    """

    def __init__(self, report_frequency=5):
        self.report_frequency = report_frequency
        self.current_epoch = 0
        self.best_map = 0

    def on_train_epoch_end(self, trainer):
        """
        Report intermediate results at the end of each epoch
        """
        self.current_epoch += 1

        try:
            # Get current metrics
            metrics = trainer.metrics
            current_map = float(metrics.box.map)

            # Update best mAP
            if current_map > self.best_map:
                self.best_map = current_map

            # Only report every report_frequency epochs to reduce overhead
            if self.current_epoch % self.report_frequency == 0:
                logger.info(f"Epoch {self.current_epoch}: mAP={current_map:.4f}, best mAP={self.best_map:.4f}")

                # Report intermediate result to NNI
                nni.report_intermediate_result(current_map)

                # Save additional metrics for analysis
                additional_metrics = {
                    'epoch': self.current_epoch,
                    'map': float(metrics.box.map),
                    'map50': float(metrics.box.map50),
                    'precision': float(metrics.box.precision),
                    'recall': float(metrics.box.recall)
                }

                # Log the metrics for debugging
                logger.info(f"Reporting to NNI: {additional_metrics}")

        except Exception as e:
            logger.error(f"Error in callback: {e}")


def train_yolo(data_yaml, params):
    """
    Train a YOLOv8 model with the given parameters
    """
    # Validate dataset before training
    data_config = validate_dataset(data_yaml)

    # Extract all parameters from params dict
    model_size = params['model_size']
    batch_size = params['batch_size']
    epochs = params['epochs']
    imgsz = params['imgsz']
    lr0 = params['lr0']
    lrf = params['lrf']
    weight_decay = params['weight_decay']
    momentum = params['momentum']
    warmup_epochs = params['warmup_epochs']
    warmup_momentum = params['warmup_momentum']
    warmup_bias_lr = params['warmup_bias_lr']
    patience = params['patience']
    augment = params['augment']
    mixup = params['mixup']
    mosaic = params['mosaic']
    degrees = params['degrees']
    translate = params['translate']
    scale = params['scale']
    shear = params['shear']
    perspective = params['perspective']
    flipud = params['flipud']
    fliplr = params['fliplr']

    logger.info(f"Training YOLOv8 model: {model_size}")
    logger.info(f"Hyperparameters: batch={batch_size}, epochs={epochs}, lr0={lr0}, imgsz={imgsz}")

    # Load the model
    try:
        model = YOLO(model_size)
        logger.info(f"Model loaded successfully: {model_size}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Create NNI callback for intermediate reporting
    nni_callback = NNIReportCallback(report_frequency=5)  # Report every 5 epochs

    # Generate a unique run name that includes the NNI trial ID
    trial_id = nni.get_trial_id()
    run_name = f"trial_{trial_id}"

    # Create a JSON file to store the mapping between NNI trial ID and run directory
    mapping_file = os.path.join('runs', 'nni_mapping.json')
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)

    # Check if mapping file exists, create it if not
    if not os.path.exists(mapping_file):
        with open(mapping_file, 'w') as f:
            json.dump({}, f)

    # Train the model
    try:
        # Train with callbacks
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
            name=run_name,
            exist_ok=True,
            callbacks=[nni_callback.on_train_epoch_end]  # Add our NNI callback
        )

        logger.info("Training completed successfully")

        # Log final metrics
        metrics = model.metrics
        logger.info(f"Final mAP50-95: {metrics.box.map}")
        logger.info(f"Final mAP50: {metrics.box.map50}")
        logger.info(f"Final precision: {metrics.box.precision}")
        logger.info(f"Final recall: {metrics.box.recall}")

        # Update the mapping between NNI trial ID and run directory
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)

        mapping[trial_id] = run_name

        with open(mapping_file, 'w') as f:
            json.dump(mapping, f)

        # Also log the path to the best model for easier retrieval
        best_model_path = os.path.join('runs', run_name, 'weights', 'best.pt')
        logger.info(f"Best model saved at: {best_model_path}")

        # Report final result to NNI
        # We maximize mAP (0-1 range, higher is better)
        final_map = float(metrics.box.map)
        nni.report_final_result(final_map)

        # Save detailed metrics for later analysis
        metrics_dict = {
            'map': final_map,
            'map50': float(metrics.box.map50),
            'precision': float(metrics.box.precision),
            'recall': float(metrics.box.recall),
            'model_size': model_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'imgsz': imgsz,
            # Add more relevant parameters
            'trial_id': trial_id,
            'run_name': run_name
        }

        # Save metrics to file
        metrics_file = os.path.join('runs', run_name, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        return metrics, best_model_path

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
        metrics, best_model_path = train_yolo(args.data_yaml, params)
        logger.info("Training and evaluation completed successfully")
        logger.info(f"Best model path: {best_model_path}")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()