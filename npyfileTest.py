import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Motor Unit Recruitment: When you perform an action like lifting your leg, your brain doesn’t send a single, uniform signal to all motor units. Instead, it recruits motor units in a specific order and pattern. Different motor units may fire at different times and rates, depending on the force required and the muscle’s fatigue state.

def loadMUAPsSample(path):
	"""Load .npy from given patrh"""
	return np.load(path)

def generate_spike_train(sample_rate, firing_profile):
    """Generate a spike train using a Poisson process based on a firing profile and sample rate"""
    return np.random.poisson(firing_profile / sample_rate)

def plot_All_muaps(muaps):
	"""Plot all muaps in the same plot for every muap collection"""

	for n in range(muaps.shape[0]):
		plt.figure(n)
		for i in range(muaps.shape[2]):
			for j in range(muaps.shape[3]):
				current_MUAP = muaps[n,0,i,j,:]
				plt.plot(current_MUAP)
	plt.show()

def plot_trapezoidal_spike_example():
	"""Plot an example of trapezoidal profile to a spike train"""
	
	duration = 5
	sample_rate = 1000

	time = np.linspace(0, duration, int(duration * sample_rate))
	profile = generate_trapezoidal_firing_profile(time, 20, 0.2)
	spike = generate_spike_train(sample_rate, profile)

	# Plot the trapezoidal firing rate profile and the spikes
	fig, ax1 = plt.subplots()

	color = 'tab:blue'
	ax1.set_xlabel('Time (s)')
	ax1.set_ylabel('Firing Rate (Hz)', color=color)
	ax1.plot(time, profile, color=color, label='Firing Rate Profile')
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	color = 'tab:red'
	ax2.set_ylabel('Amplitude', color=color)  # we already handled the x-label with ax1
	ax2.plot(time, spike, color=color, label='Spike train')
	ax2.tick_params(axis='y', labelcolor=color)

	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.title('Trapezoidal Firing Rate Profile and with spike train')
	plt.show()

def generate_trapezoidal_firing_profile(time, fire_rate=15, ramp_duration=0.2):
	"""Generate a trapezoidal firing profile based on given firing rate and duration of ramp up/down"""
	# Create a trapezoidal firing rate profile
	ramp_up_duration = int(ramp_duration * len(time)) 
	plateau_duration = int((1-(2*ramp_duration)) * len(time))
	
	# Fill out a vector with the probability of a spike being generated
	firing_profile = np.concatenate([
	    np.linspace(0, fire_rate, ramp_up_duration),
	    np.full(plateau_duration, fire_rate),
	    np.linspace(fire_rate, 0, ramp_up_duration)
	])

	return firing_profile

def convolve_spike_train_with_muap(spike_train, muap):
    """Convolve spike train with MUAP to generate EMG signal."""
    emg_signal = convolve(spike_train, muap, mode='same')
    return emg_signal

def add_noise_to_emg(emg):
	"""Add Gaussian noise to the signal."""
	noise_level = 0.005
	noise = np.random.normal(0, noise_level, len(emg))
	return emg + noise

def sum_emg_signals(emg_list):
	"""Sum up the list of simulated emg signals and returns the estimated emg signal"""
	return np.sum(emg_list,axis=0)

def plotRandomMUAP(muap):
	"""Plot a 'random' muap to see what it looks like"""
	plt.figure()
	plt.plot(muap[0,0,5,10,:])
	plt.show()

def interpolate(signal):
	"""Interpolate a given signal by factor 4"""

	num_points = len(signal)*4
	x_old = np.arange(len(signal))
	x_new = np.linspace(0, len(signal) - 1, num_points)
	new_signal = np.interp(x_new, x_old, signal)

	return new_signal
	
def generateEMG(muaps, muapSampleNumber, duration, sample_rate, spike_train_firing_rate):
	"""Simulate an EMG signal based on Motor Unit Action Potentials (MUAPs)"""
	EMGlist = []

	# for each muap in sampleNumber MUAP
	for i in range(muaps.shape[2]): #range(6,9):
		for j in range(muaps.shape[3]): #range(0,3): 
			
			# Retrieve current MUAP
			current_MUAP = muaps[muapSampleNumber,0,i,j,:]

			# Generate a spike train
			time = np.linspace(0, duration, int(duration * sample_rate))

			firing_profile = generate_trapezoidal_firing_profile(time, spike_train_firing_rate, ramp_duration=0.3)

			current_spike = generate_spike_train(sample_rate,firing_profile)

			# Convolve spike train and current MUAP
			current_EMG = convolve_spike_train_with_muap(current_spike,current_MUAP)

			# Add noise
			#current_emg_with_noise = add_noise_to_emg(current_EMG)

			# Add to list of EMG signals
			EMGlist.append(current_EMG)

	return EMGlist,time


muaps = loadMUAPsSample(r"C:\Users\Morten\Documents\GitHub\BioMime\res\muaps_sample.npy")

#plot_Random_muap(muaps)
#plot_All_muaps(muaps)
#plot_trapezoidal_spike_example()

duration = 4 # seconds
sample_rate = 1000 # Hz
spike_train_firing_rate = 20 # Hz (estimated to be between ~8 and ~30)
emgs,time = generateEMG(muaps,0,duration,sample_rate,spike_train_firing_rate)

summed_emgs = sum_emg_signals(emgs)


plt.figure()
plt.plot(time,summed_emgs)
plt.title("Generated EMG from BioMime muaps")
plt.xlabel("Duration (seconds)")
plt.ylabel("Amplitude (au)")
plt.show()
