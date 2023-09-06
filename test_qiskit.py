import pennylane as qml
import qiskit
from pennylane_qiskit import QiskitDevice

# 创建一个 QiskitDevice 对象，使用 Qiskit Aer 的 statevector_simulator 后端，指定四个量子比特
dev = QiskitDevice("statevector_simulator", wires=4)

# 定义一个哈密顿量，它是一个四量子比特的 XY 模型
coeffs = [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5]
ops = [
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliY(1),
    qml.PauliX(2) @ qml.PauliX(3),
    qml.PauliY(2) @ qml.PauliY(3),
    qml.PauliX(1) @ qml.PauliX(2),
    qml.PauliY(1) @ qml.PauliY(2),
    qml.PauliX(0) @ qml.PauliX(3),
    qml.PauliY(0) @ qml.PauliY(3),
]
hamiltonian = qml.Hamiltonian(coeffs, ops)

# 创建一个代价函数，它会调用 NumPyMinimumEigensolver 来求解最小本征值和本征态
cost_fn = qml.ExpvalCost(lambda params, wires: None, hamiltonian, dev)

# 调用代价函数，得到最小本征值和本征态
result = cost_fn()
print("Minimum eigenvalue: ", result[0])
print("Corresponding eigenstate: ", result[1])
