import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Motor Unit Recruitment: When you perform an action like lifting your leg, your brain doesn’t send a single, uniform signal to all motor units. Instead, it recruits motor units in a specific order and pattern. Different motor units may fire at different times and rates, depending on the force required and the muscle’s fatigue state.

def loadMUAPsSample(path):
	return np.load(path)

def generate_spike_train(duration, rate):
    """Generate a spike train using a Poisson process."""
    time = np.arange(0, duration, 0.4375)  # time vector in seconds
    spike_train = np.random.poisson(rate * 0.001, len(time))
    return spike_train

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
	return np.sum(emg_list,axis=0)

def plotRandomMUAP(muap):
	plt.figure()
	plt.plot(muap[0,0,5,10,:])
	plt.show()

def interpolate(signal):

	num_points = len(signal)*4
	x_old = np.arange(len(signal))
	x_new = np.linspace(0, len(signal) - 1, num_points)
	new_signal = np.interp(x_new, x_old, signal)

	return new_signal
	


def generateEMG(muaps, sampleNumber):

	EMGlist = []
	# for each muap in sampleNumber MUAP
	for i in range(muaps.shape[2]):
		for j in range(muaps.shape[3]):
			
			# Retrieve current MUAP
			current_MUAP = muaps[sampleNumber,0,i,j,:]

			# Generate a spike train
			duration = 42
			rate = 10
			current_spike = generate_spike_train(duration, rate)

			# Convolve spike train and current MUAP
			current_EMG = convolve_spike_train_with_muap(current_spike,current_MUAP)
			
			i_current_EMG = interpolate(current_EMG)

			# Add noise
			current_emg_with_noise = add_noise_to_emg(i_current_EMG)

			# Add to list of EMG signals
			EMGlist.append(current_emg_with_noise)
	
	# return EMGs
	return EMGlist


muaps = loadMUAPsSample(r"C:\Users\Morten\Documents\GitHub\BioMime\res\muaps_sample.npy")

#plotRandomMUAP(muaps)

emgs = generateEMG(muaps,0)

summed_emgs = sum_emg_signals(emgs)

plt.plot(summed_emgs)
plt.show()

notEndYet = 1