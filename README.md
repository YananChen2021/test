Communication-Efficient Cross-Modal Few-Shot Learning



## Repository layout

```text
.
├── configs/train.json       # Representation-training configuration
├── ours/
│   ├── agent.py             # Training controllers and query-state construction
│   ├── checkpoint.py        # Portable checkpoint loading and inference policy
│   ├── data.py              # Paired-image loading and episode sampling
│   ├── engine.py            # Cost profiling
│   ├── model.py             # Dual DeiT-Tiny encoders and fusion network
│   └── utils.py             # Seeds, configuration and checkpoint utilities
├── weights/ours_table1.pt   # Trained model and final Q-network
├── train.py                # Representation learning
├── train_policy.py         # Final policy learning with frozen representations
├── export_model.py         # Strip training state from a final checkpoint
├── evaluate.py             # Evaluate
├── requirements.txt
└── pyproject.toml          # Python formatting configuration
```

## Installation

Use Python 3.9 or newer in a virtual environment. Install a PyTorch build compatible with your hardware, then install the requirements:

```bash
python -m pip install -r requirements.txt
```

The runtime dependencies are PyTorch, NumPy and Pillow. CUDA is recommended for training and the full evaluation. CPU execution is supported. Run all commands below from the repository root. 


## Evaluate the released checkpoint

```bash
python evaluate.py --data-root /path/to/DATA_ROOT --device auto --output outputs/table1
```


## Train the proposed method

### 1. Learn the representation

```bash
python train.py --config configs/train.json --root /path/to/DATA_ROOT --output runs/representation --stage all --device auto
```

### 2. Train the final query-adaptive policy

```bash
python train_policy.py --source-checkpoint runs/representation/dynamic_best.pt --root /path/to/DATA_ROOT --output runs/policy --episodes 500 --validate-every 50 --val-episodes 100 --ensemble 5 --device auto
```

### 3. Export and evaluate your trained model

```bash
python export_model.py --checkpoint runs/policy/dynamic_best.pt --output runs/ours_retrained.pt
python evaluate.py --data-root /path/to/DATA_ROOT --checkpoint runs/ours_retrained.pt --output outputs/retrained
```
