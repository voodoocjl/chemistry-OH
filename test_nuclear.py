import pennylane as qml
from pennylane import numpy as np
# from qml.qchem import nuclear_energy

symbols = ["O", "H"]
geometry = np.array([[0.0, 0.0, 0.0], [0.45, -0.1525, -0.8454]])

# symbols  = ['H', 'F']
# geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
mol = qml.qchem.Molecule(symbols, geometry, charge=1)
args = [mol.coordinates]
e = qml.qchem.nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
print(e)