# Transformer with flax nnx

## Overview
This repository contains a collection of deep learning projects and experiments built using Flax and JAX. The primary focus is on implementing, training, and evaluating Transformer architecture.

The codebase is designed to demonstrate modularity and scalability, making it easy to extend and adapt for different use cases.

## Repository Structure
```bash
flax_project/
├── model/
│   ├── __init__.py               # Module initialization
│   ├── architecture.py           # Neural network architectures 
│   └── utils.py                  # Helper functions for models
├── data/
│   ├── __init__.py               # Module initialization
│   └── loader.py                 # Data loading and preprocessing
├── training/
│   ├── __init__.py               # Module initialization
│   └── trainer.py                # Training logic and loops
├── checkpoints/
│   └── checkpoint_manager.py     # Save and load model checkpoints
├── evaluation/
│   └── evaluator.py              # Functions for model evaluation
├── tests/
│   ├── test_scaled_dot_product.py             
│   ├── test_MultiHeadAttention.py              
│   ├── test_EncoderBlock.py              
├── main.py                       # Main entry point for training and evaluation
├── config.py                     # Configuration settings
└── README.md                     # Project documentation
```

## Installation
Clone the repository:

```bash
git clone https://github.com/mohsenh17/TransformerNNX.git
cd TransformerNNX
```
Set up a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Tests

to test each part:

```bash
python -m pytest tests/*
```