#!/bin/bash
# run_pipeline.sh - Script to download data from S3 and run the NAS/tuning pipeline

set -e  # Exit on error

# Default values
S3_BUCKET=""
S3_PREFIX="dataset/"
OUTPUT_DIR="data/"
DATA_YAML="data/dataset.yaml"
AWS_PROFILE=""
AWS_REGION="us-east-1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --s3-bucket)
      S3_BUCKET="$2"
      shift 2
      ;;
    --s3-prefix)
      S3_PREFIX="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-yaml)
      DATA_YAML="$2"
      shift 2
      ;;
    --aws-profile)
      AWS_PROFILE="$2"
      shift 2
      ;;
    --aws-region)
      AWS_REGION="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$S3_BUCKET" ]; then
  echo "ERROR: S3 bucket must be specified with --s3-bucket"
  exit 1
fi

echo "=== YOLOv8 NAS/HP Tuning Pipeline ==="
echo "S3 Bucket: $S3_BUCKET"
echo "S3 Prefix: $S3_PREFIX"
echo "Output Directory: $OUTPUT_DIR"
echo "Data YAML: $DATA_YAML"
echo "AWS Region: $AWS_REGION"
if [ -n "$AWS_PROFILE" ]; then
  echo "AWS Profile: $AWS_PROFILE"
fi

# Download data from S3
echo -e "\n=== Downloading dataset from S3 ==="
PROFILE_ARG=""
if [ -n "$AWS_PROFILE" ]; then
  PROFILE_ARG="--profile $AWS_PROFILE"
fi

python s3_data_downloader.py \
  --s3-bucket "$S3_BUCKET" \
  --s3-prefix "$S3_PREFIX" \
  --output-dir "$OUTPUT_DIR" \
  --data-yaml "$DATA_YAML" \
  --region "$AWS_REGION" \
  $PROFILE_ARG

# Verify the data was downloaded successfully
if [ ! -f "$DATA_YAML" ]; then
  echo "ERROR: Data YAML file not found at $DATA_YAML after download"
  exit 1
fi

echo -e "\n=== Dataset download complete ==="

# Run the NNI experiment
echo -e "\n=== Starting NNI Experiment ==="
nnictl create --config config.yml

# Get the experiment ID
EXPERIMENT_ID=$(nnictl experiment list | grep "id" | head -n 1 | awk '{print $2}')
echo "NNI Experiment ID: $EXPERIMENT_ID"

# Wait for the experiment to complete
echo -e "\n=== Monitoring NNI Experiment ==="
echo "View the NNI web UI at http://localhost:8080"
echo "Waiting for experiment to complete..."

while true; do
  STATUS=$(nnictl experiment status $EXPERIMENT_ID | grep "Status" | awk '{print $2}')
  echo "Current status: $STATUS"
  
  if [[ "$STATUS" == "DONE" || "$STATUS" == "ERROR" || "$STATUS" == "STOPPED" ]]; then
    break
  fi
  
  sleep 30
done

echo -e "\n=== NNI Experiment completed with status: $STATUS ==="

# Export the results
echo -e "\n=== Exporting experiment results ==="
nnictl experiment export $EXPERIMENT_ID -t json > nni_results.json

# Find the best trial
echo -e "\n=== Finding the best model ==="
BEST_TRIAL_ID=$(python -c "import json; data = json.load(open('nni_results.json')); best_trial = max(data['trials'], key=lambda x: x['accuracy'] if x['accuracy'] is not None else -1); print(best_trial['id'])")
echo "Best trial ID: $BEST_TRIAL_ID"

# Find the corresponding model path
TRIAL_LOG_DIR="logs/$EXPERIMENT_ID/trials/$BEST_TRIAL_ID"
MODEL_PATH=$(grep -r "Best model saved at:" $TRIAL_LOG_DIR | awk -F': ' '{print $2}')

if [ -z "$MODEL_PATH" ]; then
  echo "WARNING: Could not find model path in logs. Using latest run directory..."
  LATEST_RUN=$(ls -t runs/train | head -n 1)
  MODEL_PATH="runs/train/$LATEST_RUN/weights/best.pt"
fi

echo "Best model path: $MODEL_PATH"

# Copy the best model to a deployment directory
echo -e "\n=== Copying best model to deployment directory ==="
mkdir -p deployment
cp $MODEL_PATH deployment/best.pt
echo "Model copied to deployment/best.pt"

# Get the parameters of the best trial
python -c "import json; data = json.load(open('nni_results.json')); best_trial = max(data['trials'], key=lambda x: x['accuracy'] if x['accuracy'] is not None else -1); print(json.dumps(best_trial['parameter'], indent=2))" > deployment/best_parameters.json
echo "Parameters saved to deployment/best_parameters.json"

echo -e "\n=== Pipeline completed successfully ==="
