[![Python 3.10](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build](https://github.com/ML-KULeuven/klay/actions/workflows/main.yml/badge.svg)](https://github.com/ML-KULeuven/klay/actions/workflows/main.yml)

# KLay


_KLay is a Python library for evaluating sparse circuits on the GPU._

To get started, install KLay using pip and check out the [documentation](https://klaycircuits.readthedocs.io/en/latest/). You can also refer to [this video](https://www.youtube.com/watch?v=ZCpDenbGoJ4) or [the paper](https://openreview.net/pdf?id=Zes7Wyif8G) for more details.

```bash
pip install klaycircuits
```

Features include:
- Evaluation in Jax or PyTorch, on CPU or GPU.
- Loading SDD and d-DNNF circuits compiled by PySDD or D4.
- Evaluation in various semirings (e.g. real, log, tropical).
- Propagating constants and merging duplicate nodes.


## 📃 Paper

If you use KLay in your research, consider citing [our paper](https://openreview.net/pdf?id=Zes7Wyif8G).

To replicate the exact results and figures of the paper, use [this code](https://github.com/ML-KULeuven/klay/tree/d3b81491c34603ba9271d25af7c789d3ba368ede).

```bibtex
@inproceedings{
    maene2025klay,
    title={{KL}ay: Accelerating Arithmetic Circuits for Neurosymbolic {AI}},
    author = {Maene, Jaron and Derkinderen, Vincent and Zuidberg Dos Martires, Pedro},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=Zes7Wyif8G}
}
```
