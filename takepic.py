# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:33:03 2019

@author: bhask
"""

import os
import numpy as np
import time
from PIL import Image


#im.show()
import numpy as np
#import matplotlib.pyplot as plt
import time

def take_pic():
    try:
         photoArr = np.load('photoArr.npy')
         print('photonum loaded')
    except:
         photoArr = np.zeros(1)
         np.save('photoArr.npy', photoArr)
         print('photonum saved')
    
    try:
        photoNum = int(photoArr[0])
        command = "fswebcam -r 1280x720 --no-banner image" + str(photoNum) +".jpg"
        os.system(command)
        print( "image" + str(photoNum) +".jpg created")

    except:
        print("Error: image" + str(photoNum) + " not saved")
    
    photoNum += 1
    photoArr = photoNum*np.ones(1)
    np.save('photoArr.npy', photoArr)
    return photoNum

def processing(im):
    (h0, w0) = im.size
    fact = 50

    t1 = time.time()
    im = im.resize((int(h0 / fact), int(w0 / fact)))
    array = np.array(im.getdata())
    (h, w) = im.size
    ## Ratios of R/G, R/B, G/B
    cosrgbmax = [0.975, 2.31, 3.07]
    cosrgbmin = [0.67, 1.43, 1.72]
    wanrgbmax = [19.62, 2.04, 0.246]
    wanrgbmin = [6.46, 1.4, 0.087]

    numcos = 0
    coshi = 0
    coswi = 0
    numwan = 0
    wanhi = 0
    wanwi = 0

    justcoswan = np.zeros((h, w))

    for i in range(w * h):
        if (np.random.random() > 0):
            r, g, b = array[i]
            ratios = [r / g, r / b, g / b]
            if sum(np.greater_equal(ratios, cosrgbmin)) == 3 and sum(np.less_equal(ratios, cosrgbmax)) == 3:
                hi = i % h
                wi = int(i / h)
                justcoswan[hi, wi] = 1
                numcos += 1
                coshi += hi
                coswi += wi
            elif sum(np.greater_equal(ratios, wanrgbmin)) == 3 and sum(np.less_equal(ratios, wanrgbmax)) == 3:
                hi = i % h
                wi = int(i / h)
                justcoswan[hi, wi] = 1
                numwan += 1
                wanhi += hi
                wanwi += wi
    if numcos is not 0:

        coshi /= numcos
        coswi /= numcos
    else:
        coshi = -1
        coswi = -1
    if numwan is not 0:
        wanhi /= numwan
        wanwi /= numwan
    else:
        wanhi = -1
        wanwi = -1
    t2 = time.time()
    print('Processing took:', t2 - t1)
    print('green coords:', round(coshi/h,3), round(coswi/w, 3))
    print('pink coords:', round(wanhi/h, 3), round(wanwi/w, 3))
    print('Mass:' + str(numwan+numcos))

for i in range(1):
    num = take_pic()
    im = Image.open("image"+str(num-1)+".jpg")
    print(im.format, im.size, im.mode)
    processing(im)
