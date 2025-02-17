# Masked Diffusion Language Model

## Overview
This project implements a **masked diffusion language model** in **PyTorch**. The model learns to predict masked tokens in an input sequence using a **Transformer-based architecture** and an iterative denoising process inspired by **diffusion models**. It can be used for **masked token prediction, text generation, and sequence refinement**.

Reference: [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)



## Features
- **Diffusion-Style Masked Language Model**: Predicts masked tokens iteratively.
- **Transformer-Based Architecture**: Uses a deep **Transformer Encoder** for token representations.
- **Support for Real Datasets**: Can train on custom datasets.
- **Advanced Sampling**: Supports **temperature scaling, top-k, and nucleus (top-p) sampling** for controlled text generation.
- **Optimized Training**: Uses **mixed-precision training (`torch.cuda.amp`)** for faster performance on GPUs.



## Installation
To set up the environment, install the required dependencies:
```bash
pip install torch numpy argparse tqdm
```

For **GPU acceleration**, ensure you have the appropriate **CUDA-enabled PyTorch** installed. Visit [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for more details.



## Usage

### Training the Model
To train the model on dummy data:
```bash
python3 main.py --mode train --epochs 10
```

To train on a **real dataset** (each line in the file should contain a sequence of tokenized numbers):
```bash
python3 main.py --mode train --dataset_file path/to/dataset.txt --epochs 10
```



### Sampling from the Model
Once trained, the model can generate text:
```bash
python3 main.py --mode sample --num_steps 10
```

To control sampling:
```bash
python3 main.py --mode sample --num_steps 10 --temperature 0.7 --top_k 50 --top_p 0.9
```
- **`temperature`**: Adjusts randomness (lower = more deterministic, higher = more creative).
- **`top_k`**: Keeps the top-k most probable tokens.
- **`top_p` (nucleus sampling)**: Selects from the smallest group of tokens whose probabilities sum to `p`.



## Model Architecture
The **Masked Diffusion Transformer** consists of:

1. **Embedding Layer**: Converts token indices into dense vector representations.
2. **Transformer Encoder**: Processes sequences using multiple **self-attention layers**.
3. **Masking Strategy**: 
   - **Training**: Randomly masks tokens for denoising prediction.
   - **Sampling**: Iteratively refines masked predictions over multiple steps.
4. **Final Linear Layer**: Computes token probabilities for masked positions.



## Performance Optimizations
- **Mixed-Precision Training**: Uses `torch.cuda.amp` for **faster training** on GPUs.
- **Gradient Scaling**: Prevents underflow during **low-precision computations**.
- **Data Parallelism**: Compatible with multi-GPU setups via **Distributed Data Parallel (DDP)**.



## References
- [Masked Diffusion Language Model (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992)
- [Attention is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
- [Denoising Diffusion Probabilistic Models (arXiv:2006.11239)](https://arxiv.org/abs/2006.11239)



## License
This project is **open-source** under the **MIT License**.
