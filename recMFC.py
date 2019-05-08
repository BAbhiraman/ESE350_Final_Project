# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:19:59 2019

@author: bhask
"""

import time
import numpy as np
from numpy import linalg
import rcpy
import rcpy.adc as adc
from micadc import record
import rcpy.servo as servo
from MFCfuncs import *
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
Nsamps = 40
Nsampsgarbo = 80
servo.enable()
rec_hitmes(Nsamps, duration, fs, ch)
rec_lefts(Nsamps, duration, fs, ch)
rec_rights(Nsamps, duration, fs, ch)
rec_ups(Nsamps, duration, fs, ch)
rec_downs(Nsamps, duration, fs, ch)
rec_garbo(Nsampsgarbo, duration, fs, ch)
