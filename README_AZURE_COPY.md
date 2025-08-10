# Azure Image Copy Script

This script copies images to Azure Blob Storage based on the deduplication CSV report.

## Overview

The script reads the deduplication report CSV file and copies images to Azure destination directories:
- **Best images** → `JohnLloyd_test/result/best/`
- **Duplicate images** → `JohnLloyd_test/result/duplicates/`

## Prerequisites

1. **Deduplication Report**: You must have run the deduplication pipeline first to generate `deduplication_report.csv`
2. **Azure Access**: Valid SAS URL configured in `azure_utils.py`
3. **Python Dependencies**: `pandas`, Azure SDK

## Quick Start

### 1. Run the Script

```bash
python copy_images_to_azure.py
```

### 2. Check the Logs

The script will:
- Read the CSV report
- Copy best images to Azure
- Copy duplicate images to Azure
- Provide progress updates
- Save detailed logs to `azure_copy.log`

## Configuration

Edit `azure_copy_config.py` to customize:

```python
# Azure destination
AZURE_DESTINATION_BASE = "JohnLloyd_test/result/"

# Report file path
REPORT_FILE_PATH = "deduplication_report.csv"

# What to copy
COPY_BEST_IMAGES = True
COPY_DUPLICATE_IMAGES = True

# Performance settings
BATCH_SIZE = 50
MAX_CONCURRENT_COPIES = 10
RATE_LIMIT_DELAY = 0.1
```

## CSV Report Format

The script expects a CSV with these columns:
- `Image Path`: Azure blob path of the image
- `Status`: Either "Best" or "Duplicate"
- `Quality Score`: Image quality score
- `Group ID`: Duplicate group identifier
- `Group Size`: Number of images in the group

## Output Structure

```
JohnLloyd_test/result/
├── best/           # Best quality images (one per group)
└── duplicates/     # Duplicate images
```

## Error Handling

- **Failed copies** are logged with details
- **Retry logic** handles temporary Azure issues
- **Rate limiting** respects Azure service limits
- **Progress tracking** shows copy status

## Monitoring

### Console Output
```
2024-01-15 10:30:00 - INFO - Starting to copy 150 best images to JohnLloyd_test/result/best/
2024-01-15 10:30:05 - INFO - ✓ Copied image1.jpg (1/150)
2024-01-15 10:30:10 - INFO - ✓ Copied image2.jpg (2/150)
...
```

### Log File
Detailed logs saved to `azure_copy.log` including:
- Copy operations
- Success/failure status
- Error details
- Performance metrics

## Troubleshooting

### Common Issues

1. **Report file not found**
   - Ensure `deduplication_report.csv` exists
   - Check file path in configuration

2. **Azure access denied**
   - Verify SAS URL is valid
   - Check container permissions

3. **Slow copy performance**
   - Reduce `MAX_CONCURRENT_COPIES`
   - Increase `RATE_LIMIT_DELAY`

### Performance Tuning

For large datasets:
```python
BATCH_SIZE = 100              # Larger batches
MAX_CONCURRENT_COPIES = 5     # Fewer concurrent operations
RATE_LIMIT_DELAY = 0.2        # More delay between operations
```

## Example Usage

### Basic Copy
```bash
python copy_images_to_azure.py
```

### Custom Configuration
```python
# In azure_copy_config.py
AZURE_DESTINATION_BASE = "my_project/results/"
REPORT_FILE_PATH = "my_report.csv"
COPY_DUPLICATE_IMAGES = False  # Only copy best images
```

## Integration with Pipeline

This script is designed to work with the deduplication pipeline:

1. **Run deduplication** → Generates CSV report
2. **Run copy script** → Copies images to Azure
3. **Verify results** → Check Azure destination

## Support

For issues or questions:
1. Check the log files for error details
2. Verify Azure configuration
3. Ensure CSV report format is correct
