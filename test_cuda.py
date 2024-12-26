import torch

def test_cuda():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    test_cuda()