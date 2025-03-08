# YOLOv8 Neural Architecture Search and Hyperparameter Tuning with S3 Integration

This project implements an end-to-end pipeline for YOLOv8 model optimization using Neural Architecture Search (NAS) and hyperparameter tuning with Microsoft NNI. The pipeline now includes S3 integration for seamless download of training data and upload of optimized models.

## Overview

The pipeline performs the following steps:
1. Download dataset from S3 bucket
2. Run NNI experiment to find optimal model architecture and hyperparameters
3. Train YOLOv8 models with different configurations
4. Track performance metrics and select the best model
5. Upload the best model back to S3

## Requirements

- Python 3.9+
- PyTorch 1.7+
- Ultralytics YOLOv8
- Microsoft NNI 2.0+
- AWS CLI and boto3
- Docker (optional)

## Dataset Structure in S3

The dataset in S3 should be organized following the YOLOv8 format:

```
s3://your-bucket/dataset/
├── dataset.yaml        # YAML file describing the dataset
├── train/
│   ├── images/         # Training images
│   └── labels/         # Training labels
├── val/
│   ├── images/         # Validation images
│   └── labels/         # Validation labels
└── test/               # Optional
    ├── images/
    └── labels/
```

The `dataset.yaml` file should contain:

```yaml
path: /path/to/dataset  # Will be automatically updated after download
train: train/images
val: val/images
test: test/images       # Optional
nc: 3                   # Number of classes
names: ['class1', 'class2', 'class3']  # Class names
```

## Setup and Usage

### Option 1: Using Docker

1. Build the Docker image:
   ```bash
   docker build -t yolov8-nas-tuning .
   ```

2. Run the container:
   ```bash
   docker run -it --gpus all -p 8080:8080 \
     -e AWS_ACCESS_KEY_ID=your_access_key \
     -e AWS_SECRET_ACCESS_KEY=your_secret_key \
     -e AWS_REGION=us-east-1 \
     yolov8-nas-tuning ./run_pipeline.sh \
     --s3-bucket your-bucket \
     --s3-prefix dataset/
   ```

### Option 2: Local Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run the pipeline script:
   ```bash
   ./run_pipeline.sh --s3-bucket your-bucket --s3-prefix dataset/
   ```

   Or run each step manually:
   ```bash
   # Download dataset
   python s3_data_downloader.py --s3-bucket your-bucket --s3-prefix dataset/ --output-dir data/

   # Run NNI experiment
   nnictl create --config config.yml
   ```

### Option 3: Using Jenkins

1. Configure Jenkins with the AWS credentials
2. Create a new pipeline job using the provided Jenkinsfile
3. Configure the environment variables in the pipeline:
   - `S3_BUCKET`: Your S3 bucket containing the dataset
   - `S3_PREFIX`: Prefix path to the dataset in your bucket
   - `AWS_CREDENTIALS_ID`: Jenkins credentials ID for AWS access

## Files Explanation

- `s3_data_downloader.py`: Script to download dataset from S3
- `train_yolo.py`: Main training script with NNI integration
- `config.yml`: NNI experiment configuration
- `search_space.json`: Hyperparameter search space definition
- `Dockerfile`: Docker configuration for containerized execution
- `Jenkinsfile`: CI/CD pipeline definition
- `run_pipeline.sh`: Convenience script to run the entire pipeline

## Customization

### Modifying the Search Space

Edit `search_space.json` to customize the hyperparameters and their ranges:

```json
{
  "model_size": {
    "_type": "choice",
    "_value": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
  },
  "batch_size": {
    "_type": "choice",
    "_value": [8, 16, 32, 64]
  },
  // Add or modify hyperparameters as needed
}
```

### Changing NNI Configuration

Modify `config.yml` to adjust experiment settings:

```yaml
maxTrialNumber: 20  # Increase for more thorough search
tuner:
  name: TPE  # Try other algorithms like BOHB, Random, etc.
```

### Advanced S3 Configuration

For complex S3 setups (cross-account access, custom endpoints, etc.), modify the S3 client initialization in `s3_data_downloader.py`.

## Best Practices

1. Start with a smaller dataset for quick iteration
2. Use smaller models (yolov8n) for initial experiments
3. Gradually increase the search space as you identify promising regions
4. Monitor experiments via the NNI web UI (http://localhost:8080)
5. Consider freezing early layers when fine-tuning