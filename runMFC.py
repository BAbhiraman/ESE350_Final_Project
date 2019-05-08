"""
Nibl

@author: Bhaskar Abhiraman and Jason Kaufmann
"""

import time
import numpy as np
from numpy import linalg
import rcpy
import rcpy.gpio as gpio
import rcpy.adc as adc
from micadc import record
from MFCfuncs import *
import os
import rcpy.servo as servo
#import sounddevice as sd

#%% Definitions
duration = 1 #seconds
fs = 6000
N = duration*fs

ch = 1

fhz_low = 300
fhz_high = 3000
nfilterbank = 26
fmels = np.linspace(M(fhz_low), M(fhz_high), nfilterbank + 2)
fhzs = Mi(fmels)

bins = np.floor((N+1)*fhzs/fs)

#make the filters
fbank = np.zeros((nfilterbank, int(np.floor(N/2+1))))
for m in range(1, nfilterbank + 1):
    f_m_minus = int(bins[m - 1])   # left
    f_m = int(bins[m])             # center
    f_m_plus = int(bins[m + 1])    # right
    fbank[m-1, :f_m_minus] = 0
    fbank[m-1, f_m_plus:] = 0
    for k in range(f_m_minus, f_m):
        fbank[m-1, k] = (k-f_m_minus)/(f_m - f_m_minus)
    for k in range(f_m, f_m_plus):
        fbank[m-1, k] = (f_m_plus - k)/(f_m_plus - f_m)
        
#%%
hitme_melcep = np.load('hitme_melcep.npy')
left_melcep = np.load('left_melcep.npy')
right_melcep = np.load('right_melcep.npy')
up_melcep = np.load('up_melcep.npy')
down_melcep = np.load('down_melcep.npy')
garbo_melcep = np.load('garbo_melcep.npy')
Nsamps = hitme_melcep.shape[1]
Nsampsgarbo = garbo_melcep.shape[1]
melceps = (hitme_melcep, left_melcep, right_melcep, up_melcep, down_melcep, garbo_melcep)

servo.enable() #allow electrons to flow to the Mbed

command_predictor = train(1, melceps, Nsamps, Nsampsgarbo) #train on the recorded samples

time.sleep(2)
#%%
cont_voice_recog(command_predictor) #start listening for voice commands
