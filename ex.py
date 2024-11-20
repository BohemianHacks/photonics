import numpy as np
from scipy.stats import poisson
from typing import List, Tuple, Optional

class PhotonicState:
    """Represents a quantum state in the photon number basis"""
    def __init__(self, num_modes: int, photon_numbers: Optional[List[int]] = None):
        self.num_modes = num_modes
        if photon_numbers is None:
            self.photon_numbers = [0] * num_modes
        else:
            assert len(photon_numbers) == num_modes
            self.photon_numbers = photon_numbers.copy()
    
    def add_photon(self, mode: int):
        """Add a single photon to specified mode"""
        if mode < self.num_modes:
            self.photon_numbers[mode] += 1
    
    def remove_photon(self, mode: int) -> bool:
        """Remove a single photon from specified mode"""
        if mode < self.num_modes and self.photon_numbers[mode] > 0:
            self.photon_numbers[mode] -= 1
            return True
        return False
    
    def get_total_photons(self) -> int:
        """Return total number of photons across all modes"""
        return sum(self.photon_numbers)

class BeamSplitter:
    """Implements a beam splitter transformation"""
    def __init__(self, transmittivity: float):
        self.t = np.sqrt(transmittivity)
        self.r = np.sqrt(1 - transmittivity)
    
    def transform(self, state: PhotonicState, mode1: int, mode2: int) -> PhotonicState:
        """Apply beam splitter transformation to two modes"""
        new_state = PhotonicState(state.num_modes, state.photon_numbers)
        n1, n2 = state.photon_numbers[mode1], state.photon_numbers[mode2]
        
        # Implement probabilistic transformation
        for _ in range(n1 + n2):
            if np.random.random() < self.t**2:
                if new_state.remove_photon(mode1):
                    new_state.add_photon(mode2)
            else:
                if new_state.remove_photon(mode2):
                    new_state.add_photon(mode1)
        
        return new_state

class PhotonDetector:
    """Simulates a photon detector with specified efficiency"""
    def __init__(self, efficiency: float, dark_count_rate: float = 0.01):
        self.efficiency = efficiency
        self.dark_count_rate = dark_count_rate
    
    def measure(self, state: PhotonicState, mode: int) -> int:
        """Measure number of photons in specified mode"""
        true_photons = state.photon_numbers[mode]
        
        # Model detection efficiency
        detected_photons = np.random.binomial(true_photons, self.efficiency)
        
        # Add dark counts
        dark_counts = np.random.poisson(self.dark_count_rate)
        
        return detected_photons + dark_counts

class PhotonicCircuit:
    """Represents a complete photonic quantum circuit"""
    def __init__(self, num_modes: int):
        self.num_modes = num_modes
        self.state = PhotonicState(num_modes)
        self.components = []
    
    def add_single_photon(self, mode: int):
        """Add a single photon source to specified mode"""
        self.state.add_photon(mode)
    
    def add_beam_splitter(self, mode1: int, mode2: int, transmittivity: float):
        """Add a beam splitter between two modes"""
        bs = BeamSplitter(transmittivity)
        self.state = bs.transform(self.state, mode1, mode2)
    
    def measure_mode(self, mode: int, detector_efficiency: float = 0.9) -> int:
        """Measure photon number in specified mode"""
        detector = PhotonDetector(detector_efficiency)
        return detector.measure(self.state, mode)
    
    def get_state(self) -> PhotonicState:
        """Return current quantum state"""
        return self.state

def simulate_hong_ou_mandel():
    """Simulate Hong-Ou-Mandel interference experiment"""
    circuit = PhotonicCircuit(2)
    
    # Initialize with one photon in each input mode
    circuit.add_single_photon(0)
    circuit.add_single_photon(1)
    
    # Add 50:50 beam splitter
    circuit.add_beam_splitter(0, 1, 0.5)
    
    # Measure both output modes
    counts_0 = circuit.measure_mode(0)
    counts_1 = circuit.measure_mode(1)
    
    return counts_0, counts_1

# Example usage and analysis
def run_hom_experiment(num_trials: int = 1000) -> Tuple[List[int], List[int]]:
    """Run multiple trials of HOM experiment"""
    mode0_counts = []
    mode1_counts = []
    
    for _ in range(num_trials):
        c0, c1 = simulate_hong_ou_mandel()
        mode0_counts.append(c0)
        mode1_counts.append(c1)
    
    return mode0_counts, mode1_counts
