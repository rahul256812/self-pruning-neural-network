# Self-Pruning Neural Network

A PyTorch implementation of a self-pruning Convolutional Neural Network (CNN) that learns to prune its own connections during training using learnable gate mechanisms.

## Overview

This project implements a CNN with prunable fully-connected layers that automatically learn to prune unnecessary connections during training. The model uses learnable gate scores that are multiplied with the weights, allowing the network to identify and remove redundant connections while maintaining accuracy on CIFAR-10 image classification.

## Features

- **Learnable Gating Mechanism**: PrunableLinear layers with sigmoid gates that learn to prune connections
- **Sparsity Regularization**: L1 penalty on gate values to encourage sparsity
- **CIFAR-10 Classification**: Trains and evaluates on the CIFAR-10 dataset
- **Visualization**: Generates histograms showing gate value distributions
- **Multiple Lambda Values**: Experiments with different sparsity regularization strengths

## Architecture

The model consists of:
- **Convolutional Backbone**: Two conv layers with BatchNorm, ReLU, and MaxPool
- **Prunable FC Layers**: Two fully-connected layers with learnable gates
- **Dropout**: 0.3 dropout rate for regularization

The PrunableLinear layer applies sigmoid gates to weights:
```
gates = sigmoid(gate_scores)
pruned_weight = weight * gates
output = linear(input, pruned_weight, bias)
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- torch
- torchvision
- matplotlib
- numpy

## Usage

Run the training script:

```bash
python train.py
```

This will:
1. Download CIFAR-10 dataset to the `data/` directory
2. Train models with different lambda values (1e-5, 5e-5, 1e-4)
3. Evaluate accuracy and sparsity for each model
4. Save gate distribution histograms to `results/`
5. Save final results to `results/results.txt`

## Results

The script outputs:
- Test accuracy for each lambda value
- Sparsity percentage (fraction of gates < 0.01)
- Gate distribution histograms
- Summary table in `results/results.txt`

## Loss Function

The total loss combines cross-entropy with sparsity regularization:

```
loss = cross_entropy_loss + λ * sum(gates)
```

Where λ controls the strength of sparsity regularization.

## Directory Structure

```
.
├── train.py           # Main training script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── data/              # CIFAR-10 dataset (auto-downloaded)
└── results/           # Output plots and results
```

## License

MIT License