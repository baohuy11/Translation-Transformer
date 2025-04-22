# Translation Transformer

A PyTorch implementation of the Transformer model for machine translation, specifically trained for English to Japanese translation.

## Project Structure

```
Translation-Transformer/
├── src/
│   ├── model.py         # Transformer model implementation
│   ├── train.py         # Training script
│   ├── translate.py     # Translation script
│   ├── config.py        # Configuration settings
│   ├── dataset.py       # Dataset handling
│   └── __init__.py
├── tokenizer/
│   ├── tokenizer_en.json  # English tokenizer
│   └── tokenizer_ja.json  # Japanese tokenizer
└── research/
    ├── attention_visual.ipynb  # Attention visualization
    └── trials.ipynb           # Research experiments
```

## Features

- Implements the original Transformer architecture from "Attention is All You Need"
- Supports English to Japanese translation
- Uses pre-trained tokenizers for both languages
- Includes attention visualization capabilities
- GPU acceleration support (CUDA and MPS)

## Requirements

- Python 3.x
- PyTorch
- HuggingFace datasets
- Tokenizers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Translation-Transformer.git
cd Translation-Transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:
```bash
python src/train.py
```

The model will be saved in the weights directory specified in the config.

### Translation

To translate a sentence:
```bash
python src/translate.py "Your English sentence here"
```

You can also use a test set index:
```bash
python src/translate.py 42  # Will use the 42nd example from the test set
```

## Model Architecture

The implementation includes:
- Multi-head attention mechanism
- Positional encoding
- Layer normalization
- Residual connections
- Feed-forward networks
- Encoder-Decoder architecture

## Configuration

Key configuration parameters (in `config.py`):
- Batch size: 8
- Number of epochs: 3
- Learning rate: 0.0001
- Sequence length: 350
- Model dimension: 512
- Number of attention heads: 8
- Dropout: 0.1

## Research

The `research` directory contains:
- Attention visualization notebooks
- Experimental trials and results

## Acknowledgments

- Based on the original Transformer paper: "Attention is All You Need" by Vaswani et al.
- Uses the Helsinki-NLP/opus-100 dataset for training