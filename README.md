# Masked Diffusion Language Model

## Overview
This project implements a masked diffusion language model in PyTorch. The model is trained to predict masked tokens in an input sequence using a Transformer-based architecture. (arXiv:2502.09992)

## Features
- implements a diffusion-style masked language model.
- uses a transformer-based architecture.
- supports training and text generation (sampling).
- utilizes bidirectional token prediction for denoising.

## Installation
To set up the environment, install the required dependencies:
```bash
pip install torch numpy argparse
```

## Usage

### Training the Model
To train the model on dummy data, run:
```bash
python3 main.py --mode train --epochs 10
```

### Sampling from the Model
After training, you can generate text by running:
```bash
python3 main.py --mode sample --num_steps 10
```

## Model Architecture
- **Embedding Layer**: Encodes tokenized input.
- **Transformer Encoder**: Processes the sequence with multiple self-attention layers.
- **Masking Strategy**: Randomly masks tokens for training and refines predictions iteratively during sampling.
- **Final Linear Layer**: Predicts token probabilities.

---

## Future Improvements
- Support for real datasets.
- More sophisticated sampling strategies.
- Optimized training with large-scale data.

## Based on

- https://arxiv.org/abs/2502.09992

## License
This project is open-source under the MIT License.
