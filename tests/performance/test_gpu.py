import torch

def test_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Test GPU computation
        x = torch.rand(5, 3)
        print("\nTensor on CPU:", x)
        x = x.cuda()
        print("Tensor on GPU:", x)
        print("GPU computation successful!")

if __name__ == "__main__":
    test_gpu() 