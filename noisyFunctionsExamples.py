# -*- coding: utf-8 -*-
"""
Examples of noisyFunctions in use
"""
import numpy as np
import matplotlib.pyplot as plt
from random import *
from pandas import Series
import noisyFunctions

#addNoiseToSignal example
series = [np.sin(i/10)+np.cos(i/120)+np.cos(i/500) for i in range(1000)]
series = Series(series)
out_series = noisyFunctions.addNoiseToSignal(series, 0, 0.5)
out_series = Series(out_series)
fig, axs = plt.subplots(2)
axs[0].plot(series, color='r', label='Signal w.o. noise')
axs[0].legend(loc='upper right')
axs[1].plot(out_series, color='b', label='Signal w. noise')
axs[1].legend(loc='upper right')
plt.show()
