# (No code changes needed, just move this file into the modules/ folder.) 

import pandas as pd
import imagehash
import numpy as np
import ast
import math
import csv
import os
import json
import logging

logger = logging.getLogger(__name__)

def save_image_info_to_csv(item, csv_path):
    """
    Save image information and features to CSV.
    
    Args:
        item (dict): Dictionary containing image information and features
        csv_path (str): Path to save the CSV file
    """
    try:
        # Prepare row data
        row = {
            'filename': item['filename'],
            'azure_path': item['azure_path'],
            'local_path': item['local_path'],
            'global_features': json.dumps(item['global_features'].tolist()),
            'local_features': json.dumps({
                'keypoints': item['local_features']['keypoints'].tolist(),
                'descriptors': item['local_features']['descriptors'].tolist()
            }),
            'wavelet_hash': json.dumps(item['wavelet_hash'].tolist())
        }
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_path)
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
    except Exception as e:
        logger.error(f"Error saving image info to CSV: {e}")
        raise

def _parse_set_field(raw):
    """
    Turn a CSV cell into a Python set.

    Handles:
      • {'a', 'b'} style strings (via ast.literal_eval)
      • 'a;b;c' fallback
      • empty / NaN cells
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
        return set()

    text = str(raw).strip()

    # Try the curly-brace literal first
    try:
        parsed = ast.literal_eval(text)
        # literal_eval could legally return something that isn't a set/list/tuple
        if isinstance(parsed, (set, list, tuple)):
            return set(parsed)
    except (ValueError, SyntaxError):
        pass  # fall through to semicolon logic


    # Fallback: split on semicolons
    return {item.strip() for item in text.split(";") if item.strip()}


def load_image_info_from_csv(csv_path):
    """
    Load image information and features from CSV.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries containing image information and features
    """
    try:
        items = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert JSON strings back to numpy arrays
                local_features_data = json.loads(row['local_features'])
                item = {
                    'filename': row['filename'],
                    'azure_path': row['azure_path'],
                    'local_path': row['local_path'],
                    'global_features': np.array(json.loads(row['global_features'])),
                    'local_features': {
                        'keypoints': np.array(local_features_data['keypoints']),
                        'descriptors': np.array(local_features_data['descriptors'])
                    },
                    'wavelet_hash': np.array(json.loads(row['wavelet_hash']))
                }
                items.append(item)
        return items
        
    except Exception as e:
        logger.error(f"Error loading image info from CSV: {e}")
        raise 