# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:53:42 2019

@author: bhask
"""

#!/usr/bin/env python3

# Contributed by BrendanSimon

# import rcpy libraries
import rcpy
import rcpy.adc as adc
import numpy as np
import time

def record(dur, fs, ch):
    N = int(dur*fs)
    out = np.zeros((int(N),1))
    #t1 = time.time()
    for i in range(N):
        out[i] = adc.get_raw(ch)
    #t2 = time.time()
    #print("with np arrays I took: ", t2-t1)
    #np.save('recording.npy', out)
    return out

if __name__ == "__main__":
    ch = 1
    record(1, 6000, ch)
