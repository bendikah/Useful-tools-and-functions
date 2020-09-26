"""
Small script for generating and plotting a noisy signal alongside the noise-free original signal
"""
import numpy as np
import matplotlib.pyplot as plt
from random import *

N = 1000 #number of time steps
x = np.zeros(N) #noisy signal to be plotted
signal = np.zeros(N) #pure, noise-free signal

for i in range(N):
    signal[i] = np.sin(i/20) #signal to be plotted as a function of i
    noise = uniform(-signal/3, signal/3)
    x[i] = signal[i] + noise[i]

t = np.arange(0, N);
fig, axs = plt.subplots(2)
fig.suptitle("Noisy signal vs pure signal")
axs[0].plot(t, x)
axs[1].plot(t, signal)

plt.show()

