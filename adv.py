import numpy as np
from scipy.stats import poisson
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class QuantumState:
    """Enhanced quantum state representation with density matrix support"""
    def __init__(self, num_modes: int, max_photons: int = 3):
        self.num_modes = num_modes
        self.max_photons = max_photons
        self.dimension = (max_photons + 1) ** num_modes
        # Initialize pure vacuum state
        self.density_matrix = np.zeros((self.dimension, self.dimension), dtype=complex)
        self.density_matrix[0, 0] = 1.0
    
    def apply_phase_shift(self, mode: int, phase: float):
        """Apply phase shift to specified mode"""
        phase_op = self._create_phase_operator(mode, phase)
        self.density_matrix = phase_op @ self.density_matrix @ phase_op.conj().T
    
    def _create_phase_operator(self, mode: int, phase: float) -> np.ndarray:
        """Create phase shift operator for specified mode"""
        single_mode_op = np.diag([np.exp(1j * phase * n) for n in range(self.max_photons + 1)])
        ops = [np.eye(self.max_photons + 1)] * self.num_modes
        ops[mode] = single_mode_op
        return block_diag(*ops)

class PhaseShifter:
    """Implements a phase shifter"""
    def __init__(self, phase: float):
        self.phase = phase
    
    def transform(self, state: QuantumState, mode: int):
        """Apply phase shift to specified mode"""
        state.apply_phase_shift(mode, self.phase)
        return state

class NoiseChannel:
    """Models various noise processes in the circuit"""
    def __init__(self, loss_rate: float = 0.1, dephasing_rate: float = 0.05):
        self.loss_rate = loss_rate
        self.dephasing_rate = dephasing_rate
    
    def apply(self, state: QuantumState, mode: int):
        """Apply noise effects to specified mode"""
        # Implement loss
        if np.random.random() < self.loss_rate:
            # Simulate photon loss
            state.density_matrix *= (1 - self.loss_rate)
        
        # Implement dephasing
        if np.random.random() < self.dephasing_rate:
            # Add random phase noise
            random_phase = np.random.uniform(0, 2 * np.pi)
            state.apply_phase_shift(mode, random_phase)
        
        return state

class EnhancedPhotonicCircuit:
    """Enhanced photonic circuit with noise modeling and analysis capabilities"""
    def __init__(self, num_modes: int, max_photons: int = 3):
        self.num_modes = num_modes
        self.max_photons = max_photons
        self.state = QuantumState(num_modes, max_photons)
        self.noise_channel = NoiseChannel()
    
    def add_phase_shifter(self, mode: int, phase: float):
        """Add a phase shifter to the circuit"""
        shifter = PhaseShifter(phase)
        self.state = shifter.transform(self.state, mode)
        # Apply noise after component
        self.state = self.noise_channel.apply(self.state, mode)
    
    def generate_random_number(self, num_bits: int = 1) -> List[int]:
        """Generate random bits using quantum measurement"""
        random_bits = []
        for _ in range(num_bits):
            # Measure photon number in random mode
            mode = np.random.randint(0, self.num_modes)
            detector = PhotonDetector(0.9)  # 90% efficiency
            result = detector.measure(self.state, mode)
            # Convert to bit based on even/odd photon number
            random_bits.append(result % 2)
        return random_bits

    def analyze_state(self) -> Dict:
        """Analyze current quantum state"""
        analysis = {}
        
        # Calculate purity
        analysis['purity'] = np.real(np.trace(self.state.density_matrix @ self.state.density_matrix))
        
        # Calculate photon number distribution
        diag = np.real(np.diag(self.state.density_matrix))
        analysis['photon_distribution'] = diag[:self.max_photons + 1]
        
        return analysis

    def visualize_state(self):
        """Create visualization of quantum state"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot photon number distribution
        photon_numbers = range(self.max_photons + 1)
        probabilities = np.real(np.diag(self.state.density_matrix))
        ax1.bar(photon_numbers, probabilities[:self.max_photons + 1])
        ax1.set_xlabel('Photon Number')
        ax1.set_ylabel('Probability')
        ax1.set_title('Photon Number Distribution')
        
        # Plot density matrix
        im = ax2.imshow(np.real(self.state.density_matrix), cmap='RdBu')
        ax2.set_title('Density Matrix (Real Part)')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        return fig

def demonstrate_qrng():
    """Demonstrate quantum random number generation"""
    circuit = EnhancedPhotonicCircuit(num_modes=2)
    
    # Add single photon and create superposition
    circuit.add_phase_shifter(0, np.pi/4)
    
    # Generate random bits
    random_bits = circuit.generate_random_number(num_bits=100)
    
    # Analyze randomness
    ones_count = sum(random_bits)
    bias = abs(0.5 - ones_count/len(random_bits))
    
    return random_bits, bias

# Example usage for quantum random number generation
random_bits, bias = demonstrate_qrng()
print(f"Generated random bits: {random_bits[:10]}...")
print(f"Bias: {bias:.3f}")
