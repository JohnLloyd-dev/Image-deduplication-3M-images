# Image Deduplication System

A high-performance, distributed image deduplication system designed to process large-scale image datasets from Azure Blob Storage using advanced deep learning techniques. The system identifies duplicate images and organizes them into separate folders: "best" images (highest quality from each duplicate group) and "duplicates" (remaining images from duplicate groups).

## Features

- **Advanced Image Deduplication**: Utilizes multiple deep learning models (CLIP, EfficientNet) for robust feature extraction and similarity comparison
- **Distributed Processing**: Parallel processing with configurable worker pools for downloads, processing, and saving
- **Azure Integration**: Seamless integration with Azure Blob Storage with connection pooling and rate limiting
- **Progress Tracking**: Built-in progress tracking and resumability for long-running operations
- **Feature Caching**: Efficient feature caching system to optimize processing
- **GPU Acceleration**: Support for GPU-accelerated processing with batch optimization
- **Rate Limiting**: Token bucket rate limiter to prevent API overload
- **Resumable Operations**: Progress tracking and checkpointing for long-running operations
- **Organized Output**: Automatically organizes results into "best" and "duplicates" folders in Azure

## Project Structure

```
.
├── modules/                    # Core functionality modules
│   ├── azure_utils.py         # Azure Blob Storage integration
│   ├── deduplication.py       # Core deduplication logic
│   ├── feature_extraction.py  # Image feature extraction
│   ├── distributed_processor.py # Distributed processing
│   ├── token_bucket.py        # Rate limiting implementation
│   └── io_utils.py           # I/O utilities
├── tests/                     # Test suite
├── pipeline.py               # Main pipeline orchestration
├── main.py                   # Entry point
├── requirements.txt          # Project dependencies
└── setup.py                  # Package setup
```

## Dependencies

- **Deep Learning**: PyTorch, CLIP, EfficientNet
- **Image Processing**: OpenCV, Pillow, Kornia
- **Cloud Storage**: Azure Storage Blob
- **Distributed Computing**: Multi-GPU processing with PyTorch
- **Feature Storage**: FAISS (CPU version)
- **Additional ML**: scikit-learn, pandas

## Output Structure

After processing, the system creates the following structure in Azure Blob Storage:

```
{original_directory}_deduplicated/
├── best/                    # Highest quality image from each duplicate group
│   ├── image1.jpg          # Best version of duplicate group 1
│   ├── image5.jpg          # Best version of duplicate group 2
│   └── ...
├── duplicates/              # All other images from duplicate groups
│   ├── image2.jpg          # Duplicate of image1.jpg
│   ├── image3.jpg          # Another duplicate of image1.jpg
│   ├── image6.jpg          # Duplicate of image5.jpg
│   └── ...
└── image_report.csv         # Detailed deduplication report
```

The CSV report contains:
- **Image Path**: Full path to the image
- **Quality Score**: Computed quality metric
- **Group ID**: Which duplicate group the image belongs to
- **Group Size**: Number of images in the duplicate group
- **Status**: "Best" or "Duplicate"
- **Avg Color Correlation**: Color similarity within the group
- **Dominant Colors**: Number of dominant colors detected

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The system can be configured through various parameters in `pipeline.py`:

- `NUM_AZURE_CONNECTIONS`: Number of Azure connections to maintain
- `MIN_DOWNLOAD_WORKERS`: Minimum number of download workers
- `MAX_DOWNLOAD_WORKERS`: Maximum number of download workers
- `NUM_PROCESS_WORKERS`: Number of processing workers
- `BATCH_SIZE`: Batch size for GPU processing
- `MAX_REQUESTS_PER_SECOND`: Rate limiting threshold

## Usage

1. Configure Azure credentials and target directory in `pipeline.py`
2. Run the pipeline:
   ```bash
   python main.py
   ```

## Architecture

The system uses a multi-stage pipeline architecture:

1. **Download Stage**: Downloads images from Azure Blob Storage
2. **Processing Stage**: Extracts features and performs deduplication
3. **Save Stage**: Saves results and maintains feature cache

Each stage operates with its own worker pool and queue system, connected through thread-safe queues.

## Performance Considerations

- Uses connection pooling for Azure operations
- Implements feature caching to reduce redundant computations
- Batch processing for GPU efficiency
- Rate limiting to prevent API overload
- Distributed processing capabilities for scalability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add appropriate license information] 