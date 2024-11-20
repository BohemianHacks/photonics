import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_moving_object(size=256, object_size=10, frames=5):
    """Create a 3D data cube with a moving object"""
    cube = np.zeros((size, size, size))
    # Place object in first few frames
    start_pos = size // 4
    for t in range(frames):
        cube[t, 
             start_pos:start_pos+object_size,
             start_pos:start_pos+object_size] = 1
    return cube

def analyze_spectrum(cube):
    """Compute 3D FFT and return spectrum characteristics"""
    spectrum = np.fft.fftn(cube)
    spectrum_magnitude = np.abs(spectrum)
    
    # Analyze frequency components
    time_freq = np.sum(spectrum_magnitude, axis=(1,2))
    spatial_freq_x = np.sum(spectrum_magnitude, axis=(0,2))
    spatial_freq_y = np.sum(spectrum_magnitude, axis=(0,1))
    
    return time_freq, spatial_freq_x, spatial_freq_y

def plot_spectrum_analysis(time_freq, spatial_freq_x, spatial_freq_y):
    """Create 3D visualization of spectrum components"""
    fig = plt.figure(figsize=(15, 5))
    
    # Time frequency component
    ax1 = fig.add_subplot(131)
    ax1.plot(time_freq[:50])  # First 50 components
    ax1.set_title('Temporal Frequency Components')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude')
    
    # Spatial frequencies
    ax2 = fig.add_subplot(132)
    ax2.plot(spatial_freq_x[:50])
    ax2.set_title('Spatial Frequency X')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude')
    
    ax3 = fig.add_subplot(133)
    ax3.plot(spatial_freq_y[:50])
    ax3.set_title('Spatial Frequency Y')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Magnitude')
    
    plt.tight_layout()
    return fig

# Generate and analyze data
cube = create_moving_object()
time_freq, spatial_freq_x, spatial_freq_y = analyze_spectrum(cube)
plot = plot_spectrum_analysis(time_freq, spatial_freq_x, spatial_freq_y)
