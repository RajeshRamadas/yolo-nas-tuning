#!/usr/bin/env python3
# s3_data_downloader.py - Script to download dataset from S3 before training

import os
import sys
import argparse
import logging
import yaml
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download dataset from S3 bucket")
    parser.add_argument('--s3-bucket', type=str, required=True,
                        help='S3 bucket name containing the dataset')
    parser.add_argument('--s3-prefix', type=str, default='dataset/',
                        help='S3 prefix/folder where dataset is stored')
    parser.add_argument('--output-dir', type=str, default='data/',
                        help='Local directory to store downloaded data')
    parser.add_argument('--data-yaml', type=str, default='data/dataset.yaml',
                        help='Path to data.yaml file (will be updated with local paths)')
    parser.add_argument('--profile', type=str, default=None,
                        help='AWS profile name to use')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region name')
    
    return parser.parse_args()

def get_s3_client(profile=None, region='us-east-1'):
    """Create an S3 client with specified profile and region"""
    try:
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
            s3_client = session.client('s3')
        else:
            # Use default credentials (environment variables or IAM role)
            s3_client = boto3.client('s3', region_name=region)
        
        return s3_client
    except Exception as e:
        logger.error(f"Error creating S3 client: {str(e)}")
        raise

def download_file(s3_client, bucket, key, local_path):
    """Download a single file from S3"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        return True
    except ClientError as e:
        logger.error(f"Error downloading file {key}: {str(e)}")
        return False

def download_directory(s3_client, bucket, prefix, local_dir):
    """Download an entire directory from S3"""
    try:
        # List all objects in the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        result_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        # Download each object
        downloaded_files = []
        for result in result_iterator:
            if "Contents" not in result:
                logger.warning(f"No objects found at s3://{bucket}/{prefix}")
                continue
                
            for obj in result["Contents"]:
                # Get the relative path
                key = obj["Key"]
                if key.endswith('/'):  # Skip directories
                    continue
                    
                # Create the local path
                rel_path = os.path.relpath(key, prefix)
                local_path = os.path.join(local_dir, rel_path)
                
                # Download the file
                if download_file(s3_client, bucket, key, local_path):
                    downloaded_files.append(local_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} files from s3://{bucket}/{prefix}")
        return downloaded_files
    except Exception as e:
        logger.error(f"Error downloading directory {prefix}: {str(e)}")
        raise

def update_data_yaml(yaml_path, output_dir):
    """Update the data.yaml file with the correct local paths"""
    try:
        # Read the data.yaml file
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update the path
        data_config['path'] = os.path.abspath(output_dir)
        
        # Make sure train/val/test paths are relative
        for key in ['train', 'val', 'test']:
            if key in data_config and data_config[key].startswith('/'):
                data_config[key] = os.path.relpath(data_config[key], data_config['path'])
        
        # Save the updated file
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Updated {yaml_path} with local paths")
        
        # Print the final configuration
        logger.info(f"Dataset configuration:")
        logger.info(f"  Path: {data_config['path']}")
        for key in ['train', 'val', 'test']:
            if key in data_config:
                logger.info(f"  {key}: {data_config[key]}")
        
        return data_config
    except Exception as e:
        logger.error(f"Error updating data.yaml: {str(e)}")
        raise

def main():
    """Main function to download dataset from S3"""
    args = parse_args()
    
    try:
        # Create S3 client
        s3_client = get_s3_client(args.profile, args.region)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Download the data.yaml file first
        yaml_key = os.path.join(args.s3_prefix, 'dataset.yaml')
        yaml_local_path = args.data_yaml
        
        if not download_file(s3_client, args.s3_bucket, yaml_key, yaml_local_path):
            logger.error(f"Failed to download dataset.yaml file. Check if it exists at s3://{args.s3_bucket}/{yaml_key}")
            sys.exit(1)
        
        # Read the data.yaml file to understand the dataset structure
        with open(yaml_local_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Download the dataset files
        for data_type in ['train', 'val', 'test']:
            if data_type in data_config:
                # Get the path relative to the dataset root
                data_path = data_config[data_type]
                
                # Construct the S3 prefix
                s3_data_prefix = os.path.join(args.s3_prefix, data_path)
                
                # Construct the local directory
                local_data_dir = os.path.join(args.output_dir, data_path)
                
                # Download the data
                logger.info(f"Downloading {data_type} data from s3://{args.s3_bucket}/{s3_data_prefix}")
                download_directory(s3_client, args.s3_bucket, s3_data_prefix, local_data_dir)
                
                # If there's a corresponding 'labels' directory, download that too for YOLOv8
                if 'images' in data_path:
                    labels_path = data_path.replace('images', 'labels')
                    s3_labels_prefix = os.path.join(args.s3_prefix, labels_path)
                    local_labels_dir = os.path.join(args.output_dir, labels_path)
                    
                    logger.info(f"Downloading {data_type} labels from s3://{args.s3_bucket}/{s3_labels_prefix}")
                    download_directory(s3_client, args.s3_bucket, s3_labels_prefix, local_labels_dir)
        
        # Update the data.yaml file with the correct local paths
        update_data_yaml(yaml_local_path, args.output_dir)
        
        logger.info("Dataset download completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
