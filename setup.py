from setuptools import setup, find_packages

setup(
    name="image_deduplication",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "paddlepaddle>=2.5.0",
        "paddleocr>=2.7.0",
        "azure-storage-blob>=12.17.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "ftfy>=6.1.1",
        "regex>=2023.6.3",
        "imagehash>=4.3.1",
    ],
    dependency_links=[
        "git+https://github.com/openai/CLIP.git"
    ],
) 