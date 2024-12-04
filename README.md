# Transformer with flax nnx

## Overview
This repository contains a collection of deep learning projects and experiments built using Flax and JAX. The primary focus is on implementing, training, and evaluating Transformer architecture.

The codebase is designed to demonstrate modularity and scalability, making it easy to extend and adapt for different use cases.

## Features
- **Custom Transformer Architecture:** Includes MultiHeadAttention, CrossMultiHeadAttention, PositionalEncoding, Decoder, and Encoder blocks.
- **Modular Codebase:** Clear separation between model definition, data loading, training, and evaluation.
- **Tests Included:** Unit tests for each core module.
- **Checkpointing:** Save and load models efficiently with robust checkpoint management.
- **Configurable Design:** Easily adjust hyperparameters, architecture, and training options via config.py.

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
│   ├── test_CrossMultiHeadAttention.py             
│   ├── test_DecoderBlock.py             
│   ├── test_EncoderBlock.py              
│   ├── test_MultiHeadAttention.py              
│   ├── test_PositionalEncoding.py        
│   ├── test_scaled_dot_product.py             
│   ├── test_TransformerEncoder.py        
│   ├── test_TransformerDecoder.py        
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