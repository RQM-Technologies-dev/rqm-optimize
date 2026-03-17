"""examples/basic_optimize.py — Minimal runnable example for rqm-optimize.

Demonstrates the core workflow: build a circuit, optimize it, inspect results.
"""

from qiskit import QuantumCircuit

from rqm_optimize import optimize

# Build a simple one-qubit circuit with multiple rotations.
qc = QuantumCircuit(1, name="example")
qc.rx(0.5, 0)
qc.ry(0.3, 0)
qc.rz(0.2, 0)
qc.h(0)
qc.rx(1.1, 0)
qc.rz(0.8, 0)

print("Original circuit:")
print(qc)
print(f"Original gate count: {len(qc.data)}")

# Optimize with metadata.
result = optimize(qc, return_metadata=True)

print("\nOptimized circuit:")
print(result.circuit)
print(f"Optimized gate count: {result.optimized_gate_count}")
print(f"Fused runs:           {result.fused_runs}")
print(f"Strategy:             {result.strategy}")
