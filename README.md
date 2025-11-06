# Arithmetic Behavior Analysis

## Requirements

- Python 3.8+
- CUDA-enabled GPU (recommended) and NVIDIA drivers
- PyTorch (CUDA build), transformers, matplotlib

Install dependencies (example for CUDA 12.1):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers matplotlib
```

## Google Colab Option

You can also use Google Colab with a T4 GPU. Select a T4 GPU as your runtime hardware accelerator. In the first cell, run:
```python
!pip install -U "transformers>=4.44" "huggingface_hub>=0.23" accelerate bitsandbytes --quiet
from huggingface_hub import login
login()
```
This will install the required libraries and prompt you to log in to Hugging Face. Then run the rest of the code as usual.