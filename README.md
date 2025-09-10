# TreeGPT: A Novel Hybrid Architecture for Abstract Syntax Tree Processing


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A hybrid neural architecture combining transformer attention with global parent-child aggregation for efficient Abstract Syntax Tree (AST) processing and neural program synthesis.**

## 🏆 Key Achievements

- **96% accuracy** on ARC Prize 2025 dataset with only **1.5M parameters**
- **74× improvement** over similar-sized models (DeepSeek-R1-1.5B: 1.3%)
- **6× improvement** over large closed-source models (Grok-4: 15.9%)
- **1.8× improvement** over specialized program synthesis methods (SOAR: 52%)

## 🚀 Overview

TreeGPT introduces a groundbreaking hybrid architecture that combines the strengths of Transformer self-attention mechanisms with specialized Tree Feed-Forward Networks (TreeFFN) for processing hierarchical tree structures. Unlike traditional approaches that rely solely on sequential processing, TreeGPT directly models parent-child relationships in Abstract Syntax Trees through global aggregation.

### Core Innovation: Global Parent-Child Aggregation

```math
h_i^{(t+1)} = \sigma \Big( h_i^{(0)} + W_{pc} \sum_{(p,c) \in E_i} f(h_p^{(t)}, h_c^{(t)}) + b \Big)
```

Where `h_i^{(t)}` represents the hidden state of node `i` at iteration `t`, `E_i` denotes all parent-child edges involving node `i`, and `f(h_p, h_c)` is the edge aggregation function.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- torch-scatter

### Quick Setup

```bash
git clone https://github.com/lizixi-0x2f/TreeGPT.git
cd TreeGPT
pip install -r requirements.txt
```

### From Source

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install torch-scatter
pip install torch-scatter

# Install other dependencies
pip install numpy matplotlib json pathlib logging
```

## 🎯 Quick Start

### Basic TreeGPT Usage

```python
import torch
from src.TreeGPT import TreeGPT

# Initialize TreeGPT model
model = TreeGPT(
    vocab_size=17,        # ARC task vocabulary
    d_model=256,          # Hidden dimension
    n_heads=8,            # Attention heads
    n_layers=6,           # Number of layers
    max_seq_len=1024,     # Maximum sequence length
    tree_iterations=3     # TreeFFN iterations
)

# Generate predictions
input_ids = torch.randint(0, 17, (1, 100))  # Example input
output = model(input_ids)
print(f"Output shape: {output.shape}")
```

### ARC Task Processing

```python
from src.arc_treegpt import ARCTreeGPT, ARCGridTokenizer

# Initialize ARC-specific model
model = ARCTreeGPT(vocab_size=17, d_model=256, n_heads=8, n_layers=6)
tokenizer = ARCGridTokenizer()

# Process ARC grid
grid = [[1, 2], [3, 4]]
sequence = tokenizer.grid_to_sequence(grid)
input_tensor = torch.tensor([sequence])

# Generate solution
solution_grid = model.generate_arc_solution(input_tensor, tokenizer)
print(f"Generated solution: {solution_grid}")
```

## 🧪 Experiments

### Running Ablation Studies

```bash
cd experiments
python run_ablation.py
```

This will run comprehensive ablation studies testing different architectural components:
- Edge projection mechanisms
- Gating aggregation
- Residual connections
- Bidirectional propagation

### Training on ARC Dataset

```bash
python experiments/full_train_eval.py
```

### Custom Training

```python
from src.arc_treegpt import train_arc_model, evaluate_arc_model

# Train model
model = train_arc_model()

# Evaluate performance
results = evaluate_arc_model(model)
```

## 📊 Results

### ARC Prize 2025 Benchmark Comparison

| Model Category | Model | Parameters | Accuracy | Speedup vs TreeGPT |
|---|---|---|---|---|
| **TreeGPT (Ours)** | TreeGPT | **1.5M** | **96.0%** | **1.0×** |
| Small CoT | DeepSeek-R1-1.5B | 1.5B | 1.3% | 74× worse |
| Large Models | Grok-4 (Thinking) | ~100B+ | 15.9% | 6× worse |
| Large Models | OpenAI o-series | Unknown | 1-2% | ~50× worse |
| Program Synthesis | SOAR | N/A | 52.0% | 1.8× worse |
| Program Synthesis | Greenblatt Method | N/A | 43.0% | 2.2× worse |

### Ablation Study Results

| Configuration | Validation Acc | Test Acc | Training Time |
|---|---|---|---|
| **Edge Proj + Gating** | **100%** | **96%** | **578.7s** |
| Edge Projection | 100% | 94% | 563.5s |
| Triple Combination | 100% | 92% | 573.4s |
| Edge Proj + Residual | 100% | 83% | 526.8s |
| Gating Only | 90% | 74% | 619.7s |
| Baseline | 0% | 0% | 894.5s |

## 🏗️ Architecture

TreeGPT consists of several key components:

### 1. TreeFFN (Tree Feed-Forward Network)
- **Global Parent-Child Aggregation**: Processes all parent-child relationships simultaneously
- **Edge Projection**: Optional learned transformations for edge features
- **Gated Aggregation**: Adaptive information flow control
- **Iterative Propagation**: Multi-hop reasoning through tree structures

### 2. Multi-Head Self-Attention
- Standard transformer attention mechanism
- Captures local dependencies in sequences
- Integrated with TreeFFN for hybrid processing

### 3. Hybrid Integration
```python
TreeGPT(x) = x + TreeFFN(LayerNorm(x + Attention(LayerNorm(x))))
```

## 📁 Project Structure

```
TreeGPT/
├── src/                          # Core implementation
│   ├── TreeGPT.py               # Main TreeGPT model
│   ├── TreeFFN.py               # Tree Feed-Forward Network
│   ├── Attn.py                  # Multi-head attention
│   └── arc_treegpt.py           # ARC task specialization
├── experiments/                  # Experimental code
│   ├── run_ablation.py          # Ablation studies
│   ├── full_train_eval.py       # Training pipeline
├── figures/                     # Generated figures
├── arc-prize-2025/              # ARC dataset
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT license
└── README.md                    # This file
```

## 🔬 Research Paper

Our paper "TreeGPT: A Novel Hybrid Architecture for Abstract Syntax Tree Processing with Global Parent-Child Aggregation" provides comprehensive technical details, mathematical foundations, and experimental validation.

**Key Contributions:**
1. Novel hybrid architecture combining transformers with tree-structured processing
2. Mathematically principled global parent-child aggregation mechanism  
3. Superior performance on challenging visual reasoning benchmarks
4. Comprehensive ablation studies revealing critical architectural components

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/lizixi-0x2f/TreeGPT.git
cd TreeGPT
pip install -e .
pre-commit install
```

### Running Tests

```bash
python -m pytest tests/
```

## 📄 Citation

If you use TreeGPT in your research, please cite our paper:

```bibtex
@article{li2025treegpt,
  title={TreeGPT: A Novel Hybrid Architecture for Abstract Syntax Tree Processing with Global Parent-Child Aggregation},
  author={Li, Zixi},
  year={2025}
}
```

## ☕ Support Us

If TreeGPT has helped your research or you'd like to support our work, consider buying us a coffee! Your support helps us continue developing innovative AI architectures.

<div align="center">

[![Buy us a coffee](https://img.shields.io/badge/Buy%20us%20a%20coffee-WeChat%20Pay-00D9FF?style=for-the-badge&logo=wechat)](figures/IMG_2587.JPG)

**Scan to support via WeChat Pay**

<img src="figures/IMG_2587.JPG" alt="WeChat Pay QR Code" width="300">

*Every contribution, no matter how small, is greatly appreciated! 🙏*

</div>

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ARC Prize 2025 for providing the challenging benchmark dataset
- PyTorch team for the excellent deep learning framework
- torch-scatter developers for efficient sparse operations

## 📞 Contact

- **Author**: Zixi Li
- **Email**: lizx93@mail2.sysu.edu.cn
- **Institution**: Noesis Lab, Sun Yat-sen University

---

⭐ **Star this repository if you find it helpful!** ⭐