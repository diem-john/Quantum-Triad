# CUDA-Q Integration

Quantum-Triad keeps the existing PennyLane QKN as the reference backend and adds
an optional NVIDIA CUDA-Q backend.

## Installation

Use Python 3.11+ and install the optional backend:

```bash
python -m pip install -r requirements-cudaq.txt
```

For NVIDIA GPU execution, use a supported Linux/WSL2 environment with a
compatible NVIDIA driver/CUDA installation.

## Usage

```python
from src.qkn_backend import create_qkn

qkn = create_qkn(
    backend="cudaq",
    n_qubits=3,
    layers=2,
    entangling_type="StronglyEntangling",
)

features = qkn.extract_temporal_quantum_features(X_seq)
```

The existing implementation remains available:

```python
qkn = create_qkn(
    backend="pennylane",
    n_qubits=3,
    layers=2,
)
```

## Backend parity

The CUDA-Q implementation mirrors the current PennyLane QKN circuit:

1. AngleEmbedding using RX rotations.
2. StronglyEntanglingLayers or BasicEntanglerLayers.
3. PennyLane-compatible entangling ranges.
4. Pauli-Z expectation values.
5. The same temporal feature tensor shapes.

Run parity tests with:

```bash
pytest tests/test_qkn_cudaq_parity.py -v
```

The next benchmarking stage should compare CPU simulation, NVIDIA GPU
simulation, kernel-evaluation throughput, memory use, and scaling with
qubit count and sample count before changing the default backend.
