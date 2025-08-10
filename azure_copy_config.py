#!/usr/bin/env python3
"""
Configuration file for Azure Image Copy Process

This file contains all the configurable settings for copying images to Azure
based on the deduplication report.
"""

# Azure Configuration
AZURE_SAS_URL = None  # Set to None to use default from azure_utils.py
AZURE_DESTINATION_BASE = "JohnLloyd_test/result/"

# Report Configuration
REPORT_FILE_PATH = "deduplication_report.csv"  # Path to the CSV report file

# Copy Configuration
COPY_BEST_IMAGES = True      # Whether to copy best images
COPY_DUPLICATE_IMAGES = True # Whether to copy duplicate images
BATCH_SIZE = 50              # Number of images to process before progress update

# Logging Configuration
LOG_LEVEL = "INFO"           # Logging level: DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True           # Whether to save logs to file
LOG_FILE_NAME = "azure_copy.log"

# Retry Configuration
MAX_RETRIES = 3              # Maximum number of retry attempts for failed copies
RETRY_DELAY_SECONDS = 2      # Delay between retry attempts

# Performance Configuration
MAX_CONCURRENT_COPIES = 10   # Maximum concurrent copy operations (Azure rate limiting)
RATE_LIMIT_DELAY = 0.1      # Delay between copy operations to respect Azure limits
