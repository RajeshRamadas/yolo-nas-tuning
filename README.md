# YOLOv8 Neural Architecture Search (NAS) & Continuous Tuning

This repository contains code for automating the tuning and deployment of YOLOv8 object detection models using Jenkins CI/CD pipeline and Microsoft NNI (Neural Network Intelligence).

## Repository Structure

```
├── train_yolo.py          # YOLOv8 training script with NNI integration
├── config.yml             # NNI experiment configuration
├── search_space.json      # Hyperparameter search space for NNI
├── Dockerfile             # Docker environment for containerized training
├── Jenkinsfile            # CI/CD pipeline definition
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

- Jenkins server with Pipeline, Git, and Docker plugins
- GPU-enabled server for training
- Python 3.7+ environment
- Object detection dataset in YOLOv8 format

### Dataset Preparation

Prepare your dataset in YOLOv8 format with a `dataset.yaml` file:

```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

nc: 80  # number of classes
names: ["person", "bicycle", ...]  # class names
```

Place this file in the `data/` directory.

### Jenkins Pipeline Setup

1. Install Jenkins and the required plugins
2. Create a new Pipeline job in Jenkins
3. Set up repository access in Source Code Management
4. Configure build triggers (e.g., Poll SCM)
5. Set Pipeline definition to "Pipeline script from SCM"
6. Specify the repository URL and branch
7. Set the script path to "Jenkinsfile"

## How It Works

1. The pipeline clones the repository
2. Sets up a Python environment with the required dependencies
3. Starts an NNI experiment to search for optimal YOLOv8 configurations
4. Monitors the training progress
5. Selects the best model based on performance
6. Deploys the model to the deployment directory

## Customization

- Modify `search_space.json` to adjust the hyperparameter search space
- Adjust `config.yml` to change NNI experiment settings
- Update `train_yolo.py` for custom training logic
- Edit the `Jenkinsfile` to change the CI/CD pipeline behavior

## Usage

1. Push changes to the repository to trigger the pipeline automatically
2. Monitor the training progress in the Jenkins console output
3. View detailed metrics in the NNI dashboard
4. Access the best model in the deployment directory after completion

## License

[MIT License](LICENSE)