# ML From Scratch

This repository contains implementations of machine learning algorithms from scratch.

## Structure

- **data/**: Central Data Storage
- **autograd/**: Scalar-based Engine (Value class, Neuron/Layer logic)
- **tensor_autograd/**: Tensor-based Engine (supports `@` matmul, `.T` transpose, and multidimensional gradients)
- **trees/**: Tree-based Models (Decision Tree, Random Forest, XGBoost)
- **linear_models/**: Linear Regression, Logistic Regression, SVM
- **unsupervized/**: Singular Value Decomposition (SVD) and Data Compression
- **probabilistic/**: MCMC/Sampling
- **utils/**: Shared Helpers

## Installation

```bash
pip install -r requirements.txt
```
