import qiskit
import numpy as np


class QuantumCircuit:
    """ This class is to function the circuit"""
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        self.backend = backend
        self.shots = shots
        all_qubits = [i for i in range(n_qubits)]

        self._circuit.h(all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

    def run(self, thetas):
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots,
                             parameter_binds=[{self.theta: theta} for theta in thetas])
        counts = job.result().get_counts()
        values = np.array(list(counts.values()))
        states = np.array(list(counts.keys())).astype(float)
        probabilities = values / self.shots
        expectations = np.sum(probabilities * states)
        return np.array([expectations])
