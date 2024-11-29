import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Step 1: Generate Multiple Spike Trains
def generate_spike_train(duration, rate):
    """Generate a spike train using a Poisson process."""
    time = np.arange(0, duration, 0.001)  # time vector in seconds
    spike_train = np.random.poisson(rate * 0.001, len(time))
    return time, spike_train

# Step 2: Simulate Multiple MUAPs
def simulate_muap(frequency, decay):
    """Create a simple MUAP template."""
    t = np.linspace(0, 0.01, 100)  # 10 ms duration
    muap = np.exp(-t / decay) * np.sin(2 * np.pi * frequency * t)  # Example MUAP waveform
    return muap

# Step 3: Convolve Each Spike Train with Its Corresponding MUAP
def convolve_spike_train_with_muap(spike_train, muap):
    """Convolve spike train with MUAP to generate EMG signal."""
    emg_signal = convolve(spike_train, muap, mode='same')
    return emg_signal

# Step 4: Sum the Resulting EMG Signals
def sum_emg_signals(emg_signals):
    """Sum multiple EMG signals."""
    return np.sum(emg_signals, axis=0)

# Step 5: Add Noise (Optional)
def add_noise(signal, noise_level):
    """Add Gaussian noise to the signal."""
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

# Parameters
duration = 1.0  # 1 second
rates = [10, 30]  # Spike rates for two motor units
frequencies = [50, 70]  # Frequencies for two MUAPs
decays = [0.002, 0.003]  # Decay constants for two MUAPs
noise_level = 0.1

# Generate spike trains and MUAPs
time, spike_train1 = generate_spike_train(duration, rates[0])
_, spike_train2 = generate_spike_train(duration, rates[1])
muap1 = simulate_muap(frequencies[0], decays[0])
muap2 = simulate_muap(frequencies[1], decays[1])

# Convolve spike trains with MUAPs
emg_signal1 = convolve_spike_train_with_muap(spike_train1, muap1)
emg_signal2 = convolve_spike_train_with_muap(spike_train2, muap2)

# Sum the resulting EMG signals
emg_signal = sum_emg_signals([emg_signal1, emg_signal2])

# Add noise
emg_signal_noisy = add_noise(emg_signal, noise_level)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(time, spike_train1, label='Spike Train 1')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, spike_train2, label='Spike Train 2')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(muap1, label='MUAP 1')
plt.plot(muap2, label='MUAP 2')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time, emg_signal_noisy, label='Simulated EMG with Noise')
plt.legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
