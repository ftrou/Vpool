# Efficient Inference Offload

This repository demonstrates an optimized inference pipeline for large language models on consumer-grade GPUs (e.g., a 4070 Super with 12 GB VRAM) using offloading and mixed precision.

## Features

- **Device Offloading:** Uses `device_map="auto"` to assign only necessary parts of the model to the GPU.
- **Mixed Precision Inference:** Uses `torch.cuda.amp.autocast()` to reduce memory usage and speed up inference.
- **Performance Monitoring:** Logs GPU VRAM and system RAM usage.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- psutil

Install dependencies via:

```bash
pip install -r requirements.txt
# Vpool
