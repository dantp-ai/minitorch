<p align="center">
  <img src="minitorch-logo.svg" alt="MiniTorch" width="440">
</p>

> A deep learning framework built from scratch — reverse-mode autodiff, n-dimensional tensors, and a neural-network stack, with NumPy, Numba-parallel CPU, and CUDA backends.

![Python](https://img.shields.io/badge/python-3.12-blue)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?logo=numpy)
![Numba](https://img.shields.io/badge/Numba-JIT%20%2B%20CUDA-00A3E0)
![Tests](https://img.shields.io/badge/tests-200%2B%20passing-success)

MiniTorch is a minimal, PyTorch-style autodiff engine implemented end-to-end in
Python — no `torch` under the hood. It rebuilds the machinery a modern deep
learning library hides behind the abstractions: a computation graph that records
operations, reverse-mode automatic differentiation that walks it backward, and
tensor ops that run on three interchangeable backends. The same model code
trains on a pure-Python tensor, a Numba-JIT parallel CPU kernel, or a CUDA GPU
kernel by swapping a single backend object.

---

## Highlights

- **Reverse-mode autodiff** — a `Variable`/`Function` graph with `forward`/`backward`
  for every op, topological-sort backprop, and gradient accumulation. Verified
  against numerical central-difference gradients.
- **N-dimensional tensors** — strided storage, broadcasting, views, perm/ reshape,
  and zero-copy indexing, all differentiable.
- **Three execution backends** — the same operators compiled as:
  pure Python, **Numba `@njit(parallel=True)`** CPU kernels, and **CUDA** kernels
  with shared-memory tiled matrix multiply.
- **Neural-network stack** — 1D/2D convolution, max/avg pooling, softmax &
  log-softmax (log-sum-exp stable), ReLU/sigmoid, dropout, and an SGD optimizer.
- **Tested like a library** — 200+ property-based tests (Hypothesis) covering
  operators, gradients, broadcasting, and every layer.

---

## How it fits together

The codebase is layered, each module builds strictly on the one below it.

| Layer | Module | What it provides |
|------|--------|------------------|
| **Math core** | `operators.py` | Elementary scalar functions + functional `map`/`zip`/`reduce` |
| **Scalar autodiff** | `scalar.py`, `autodiff.py` | Single-value variables, the backprop engine, chain rule |
| **Tensors** | `tensor_data.py`, `tensor.py`, `tensor_ops.py` | Strided n-d arrays, broadcasting, differentiable tensor ops |
| **Speed** | `fast_ops.py`, `fast_conv.py`, `cuda_ops.py` | Numba-parallel CPU and CUDA backends + fast matmul |
| **Networks** | `nn.py`, `module.py`, `optim.py` | Conv, pooling, softmax, dropout, `Module`, SGD |

```
input → Tensor (records op on the graph) → forward → loss
                                              │
loss.backward() ← topological sort ← chain rule ← stored contexts
```

---

## Quickstart

Requires **Python 3.12**. Recommended using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Run the test suite:

```bash
pytest tests/                 # everything
pytest tests/ -m task2_3      # one task group (markers task0_1 … task4_4)
```

> CUDA kernels compile lazily; the GPU test groups (`task3_3`, `task3_4`) are
> skipped automatically when no CUDA device is present.

---

## Train something

The `project/` directory contains runnable demos that train models built on the
engine:

| Script | Task |
|--------|------|
| `project/run_tensor.py` | Binary classification on toy datasets (xor, circle, spiral) |
| `project/run_fast_tensor.py` | Same, on the Numba-parallel backend |
| `project/run_mnist_multiclass.py` | MNIST digit classification with a small ConvNet |
| `project/run_sentiment.py` | Sentence sentiment with embeddings + 1D conv |

```bash
python project/run_tensor.py
```

---

## Tech stack

`Python 3.12` · `NumPy` · `Numba` (JIT + CUDA) · `Hypothesis` · `pytest`

---

## Credits

Built on the excellent [**MiniTorch**](https://minitorch.github.io/) teaching
framework by Sasha Rush (Cornell Tech). MiniTorch provides the scaffolding,
tests, and project structure; the engine internals in this repository —
autodiff, tensor ops, the Numba/CUDA kernels, convolutions, and the NN layers —
are my own implementation.
