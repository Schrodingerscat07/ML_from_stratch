# Learning Guide: Mastering ML From Scratch

This guide explains the repository from the ground up. Read the code while you read this guide. The
goal is to build intuition for *every* line: what it stores, why it exists, and how it contributes
to the algorithm.

## How to Use This Guide

1. **Open the file mentioned in each section** and read along.
2. **Pause at each bullet** and verify you can connect the idea to the exact line of code.
3. **Re-run the mental model**: if you can predict what the next line should be, you understand it.
4. **Do the micro-exercises** to cement intuition (even tiny ones).

## Prerequisites (Know or learn while reading)

- Python basics (classes, `__call__`, operator overloading)
- NumPy array operations (`@`, `.T`, broadcasting)
- Calculus basics (derivatives, chain rule)
- Linear algebra (vectors, matrices, dot products)

## Study Order (Scratch-by-scratch)

1. `autograd/engine.py` — scalar autograd engine
2. `autograd/nn.py` — neurons, layers, and MLPs built on the scalar engine
3. `tensor_autograd/tensor.py` — tensor autograd engine
4. `tensor_autograd/ops.py` — pure NumPy ops (no gradients)
5. `linear_models/linear_regression.py` — gradient descent + tracking history
6. `trees/decision_tree.py` — split rules and information gain
7. `trees/random_forest.py` — bagging + voting
8. `trees/xgboost.py` — gradient boosting via residuals
9. `unsupervized/svd.py` — unsupervised learning via power iteration and deflation (folder name uses this spelling)
10. `tests/test_tensor.py` — how gradients are validated
11. `linear_models/logistic_regression.py`, `probabilistic/mcmc.py`, `utils/*` — currently placeholders

---

## 1) `autograd/engine.py`: Scalar Autograd Engine

### Big Idea
Each `Value` is a single number **plus** the history of how it was computed. We store that history
so we can apply the chain rule backward.

### What to notice line-by-line

- **`self.data`**: the numeric scalar value.
- **`self.grad`**: the derivative of the final output with respect to this value.
- **`self._prev`**: the *parents* of this node in the computation graph.
- **`self._backward`**: a function that knows how to push gradients to those parents.
- **`self._op`**: a string tag like `"+"`, `"*"` so you can inspect what created the node.
- **Operator overloads** (`__add__`, `__mul__`, etc.) always:
  1. **Compute the new value** (`out.data`)
  2. **Store parents** (`(self, other)`)
  3. **Define local gradient rule** in `out._backward`

### Example: Addition (`__add__`)
If `out = a + b`, then:
- `out.data = a.data + b.data`
- `dout/da = 1` and `dout/db = 1`
So the backward pass does:
```
a.grad += out.grad
b.grad += out.grad
```

### Example: Multiplication (`__mul__`)
If `out = a * b`, then:
- `out.data = a.data * b.data`
- `dout/da = b` and `dout/db = a`
So the backward pass does:
```
a.grad += b.data * out.grad
b.grad += a.data * out.grad
```

### `tanh()` and `exp()`
- The forward uses the math formula.
- The backward multiplies by the derivative:
  - `d/dx tanh(x) = 1 - tanh(x)^2`
  - `d/dx exp(x) = exp(x)`

### `backward()`
1. **Build topological order**: list of nodes so parents appear before children.
2. **Reset grads** to `0.0`, so multiple backward calls don’t accumulate.
3. **Seed gradient**: `self.grad = 1.0` for the final output.
4. **Reverse traversal**: apply each node’s `_backward` to push gradients to parents.

### Micro-exercises
- Manually compute `((a + b) * c).tanh()` and predict which `_backward` functions will run.
- Change `a.data` and predict how gradients will change.

---

## 2) `autograd/nn.py`: Neurons, Layers, MLP

### Neuron
- `self.w`: list of `Value` weights.
- `self.b`: bias term (also a `Value`).
- `__call__(x)`:
  - Computes weighted sum: `sum(w_i * x_i)` → this builds a graph of multiplies and adds.
  - Adds bias.
  - Applies `tanh` activation.

### Layer
- A layer is just a list of neurons.
- `__call__` feeds the same input to each neuron and collects outputs.

### MLP
- `sz = [nin] + nouts` defines layer sizes.
- Each `Layer` connects size `sz[i] -> sz[i+1]`.
- `__call__` feeds output of one layer into the next.

### Micro-exercises
- Track how many `Value` nodes get created for a single forward pass of a 2-2-1 MLP.
- Replace `tanh` with `exp` and predict what breaks (saturation, gradient explosion).

---

## 3) `tensor_autograd/tensor.py`: Tensor Autograd Engine

### Big Idea
Same concept as `Value`, but `data` is a NumPy array and gradients are arrays of the same shape.

### Key Attributes
- `self.data`: NumPy array.
- `self.grad`: NumPy array initialized with zeros.
- `self._prev`: parents in the graph.

### Important Operations
- **`__add__` / `__mul__`**: element-wise operations with element-wise gradients.
- **`__matmul__`**:
  - Forward: `self.data @ other.data`
  - Backward:
    - `dL/dA = dL/dC @ B^T`
    - `dL/dB = A^T @ dL/dC`
- **`transpose` / `.T`**:
  - Forward: `A.T`
  - Backward: `grad.T`

### `backward()`
The output gradient is initialized to **ones** (same shape as output), then we reverse through the
graph and apply each stored `_backward`.

### Micro-exercises
- Compute a `2x2 @ 2x2` matmul and verify the shapes in the backward pass.
- Change a single element in `data` and see which `grad` entries change.

---

## 4) `tensor_autograd/ops.py`: Pure NumPy Ops

These functions are **stateless helpers**. They do *not* build a graph or store gradients; they just
return `numpy` computations. Use them when you want raw numeric results without autograd.

### Micro-exercises
- Compare `Tensor` operations with these pure ops on the same inputs.
- Note the missing gradients and explain why.

---

## 5) `linear_models/linear_regression.py`: Gradient Descent

### Big Idea
We minimize mean squared error by repeatedly nudging weights and bias in the direction of the
negative gradient.

### Key Steps in `fit`
1. **Initialize** weights `w` and bias `b` to zeros.
2. **Predict**: `y_pred = X @ w + b`
3. **Loss**: mean squared error `(y - y_pred)^2`
4. **Gradients**:
   - `dw = (2/n) * X.T @ (y_pred - y)`
   - `db = (2/n) * sum(y_pred - y)`
5. **Update**:
   - `w -= lr * dw`
   - `b -= lr * db`
6. **Store history** for visualization.

### Micro-exercises
- Derive `dw` and `db` by hand for a single feature.
- Explain why `loss_history` should decrease when learning rate is reasonable.

---

## 6) `trees/decision_tree.py`: Decision Trees

### Big Idea
At each node, pick a feature and threshold that **reduces uncertainty the most**.

### Key Pieces
- **Node**: stores `feature`, `threshold`, `left`, `right`, or a `value` if leaf.
- **Stopping criteria**:
  - max depth reached
  - only one class left
  - too few samples
- **Information gain**:
  - Parent loss minus weighted child losses.
  - For classification: **entropy**
  - For regression: **variance**

### Micro-exercises
- Compute entropy of `[0, 0, 1, 1, 1]`.
- Why does a split with an empty left or right return 0 gain?

---

## 7) `trees/random_forest.py`: Random Forest

### Big Idea
Train many different trees on bootstrap samples and **combine** their predictions.

### Key Steps
- **Bootstrap**: sample with replacement to create diverse datasets.
- **Fit**: each tree trains independently.
- **Predict**:
  - Classification: majority vote
  - Regression: average

### Micro-exercises
- Explain how bootstrap sampling reduces correlation between trees.
- Track the shape of `tree_preds` and why we swap axes.

---

## 8) `trees/xgboost.py`: Gradient Boosting (Simplified)

### Big Idea
Each new tree models the **residual errors** of the current ensemble.

### Steps
1. Start with a base prediction (mean of `y`).
2. Compute residuals: `y - current_prediction`.
3. Fit a tree to residuals.
4. Update prediction by adding `learning_rate * tree_prediction`.

### Micro-exercises
- Show how residuals shrink after each step for a tiny dataset.
- Explain why this is called “gradient boosting”.

---

## 9) `unsupervized/svd.py`: Singular Value Decomposition (Unsupervised Learning; folder name uses this spelling)

### Big Idea
SVD factorizes `A` into `U Σ V^T`. This implementation uses **power iteration** to find one
singular vector at a time, then **deflates** the matrix.

### Steps
1. Start with a random vector `v`.
2. Repeatedly apply `A^T A v` and normalize (power iteration).
3. Compute `sigma = ||A v||`.
4. Compute `u = A v / sigma`.
5. Remove the rank-1 component from `A` (deflation).

### Micro-exercises
- Explain why `A^T A` gives right singular vectors.
- What happens if `sigma` is near zero?

---

## 10) `tests/test_tensor.py`: Verifying Gradients

The tests:
- Build small matrices.
- Call `.backward()` to fill gradients.
- Compare to manually computed expected gradients.

### Micro-exercises
- Derive the expected gradients in the tests without looking at the code.
- Change the shapes and predict how the expected gradients change.

---

## 11) Placeholders (Work-in-progress files)

These files currently contain imports or are empty. Treat them as **exercise slots**:

- `linear_models/logistic_regression.py`
- `probabilistic/mcmc.py`
- `utils/data_loader.py`
- `utils/plotting.py`

### Suggested Next Steps
- Implement logistic regression with a sigmoid and binary cross-entropy.
- Add a basic Metropolis-Hastings sampler in `mcmc.py`.
- Add CSV loading utilities in `data_loader.py`.
- Add plotting helpers for loss curves in `plotting.py`.

---

## Mastery Checklist (Self-Test)

- [ ] I can explain every attribute in `Value` and `Tensor` without looking.
- [ ] I can derive the gradient rules used in the code.
- [ ] I can trace a forward and backward pass by hand.
- [ ] I can explain how trees pick a split and why forests vote.
- [ ] I can explain why SVD uses power iteration and deflation.

If every box is checked, you truly understand the repository.
