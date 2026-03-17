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
qc.s(0)
qc.t(0)

print("Original circuit:")
print(qc)

# Optimize with metadata (default compact U basis).
result = optimize(qc, return_metadata=True)

print("\nOptimized circuit (U basis):")
print(result.circuit)
print(f"original gates:  {result.original_gate_count}  →  optimized gates:  {result.optimized_gate_count}")
print(f"original depth:  {result.original_depth}  →  optimized depth:  {result.optimized_depth}")
print(f"original 1q:     {result.original_1q_gate_count}  →  optimized 1q:     {result.optimized_1q_gate_count}")
print(f"fused runs:      {result.fused_runs}")
print(f"strategy:        {result.strategy}")

# Same circuit using IBM-native basis (rz + sx).
result_ibm = optimize(qc, native_basis="ibm", return_metadata=True)

print("\nOptimized circuit (IBM native basis: rz + sx):")
print(result_ibm.circuit)
print(f"optimized gates: {result_ibm.optimized_gate_count}")
