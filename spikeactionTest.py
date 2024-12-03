import numpy as np
import matplotlib.pyplot as plt

def generate_muap(duration=0.042, sampling_rate=1000):
    t = np.linspace(0, duration, int(duration * sampling_rate))
    muap = np.exp(-((t - duration / 2) ** 2) / (2 * (duration / 10) ** 2))
    return muap

def generate_spike_train(firing_rate, duration=4, muap_duration=0.042):
    # Define the time vector
    t = np.linspace(0, duration, int(duration * 1000))  # 1 ms time steps

    # Create a trapezoidal firing rate profile
    ramp_up_duration = int(0.2 * len(t))  # 20% of the total duration
    plateau_duration = int(0.6 * len(t))  # 60% of the total duration
    ramp_down_duration = len(t) - ramp_up_duration - plateau_duration

    firing_profile = np.concatenate([
        np.linspace(0, firing_rate, ramp_up_duration),
        np.full(plateau_duration, firing_rate),
        np.linspace(firing_rate, 0, ramp_down_duration)
    ])

    # Generate the spike train
    spike_train = np.random.poisson(firing_profile / 1000)

    # Generate the MUAP
    muap = generate_muap(duration=muap_duration)

    # Convolve the spike train with the MUAP
    convolved_signal = np.convolve(spike_train, muap, mode='same')

    # Normalize the convolved signal so it doesn't exceed 1 in amplitude
    #convolved_signal = convolved_signal / np.max(convolved_signal)

    return t, firing_profile, convolved_signal

# Example usage
firing_rate = 11  # Example firing rate in Hz
t, firing_profile, convolved_signal = generate_spike_train(firing_rate)

# Plot the trapezoidal firing rate profile and the convolved signal
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Firing Rate (Hz)', color=color)
ax1.plot(t, firing_profile, color=color, label='Firing Rate Profile')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Amplitude', color=color)  # we already handled the x-label with ax1
ax2.plot(t, convolved_signal, color=color, label='Convolved Signal')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Trapezoidal Firing Rate Profile and Convolved Spike Train Signal')
plt.show()
