# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:26:36 2019

@author: bhask
"""

import sounddevice as sd
import matplotlib.pyplot as plt
import time
import scipy.fftpack as fft
import numpy as np

def M(f):
    return 1125.0*np.log(1.0+f/700.0)

def Mi(m):
    return 700.0*(np.exp(m/1125.0)-1.0)


duration = 1 #seconds
fs = 16000
N = duration*fs
myrecording = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
time.sleep(duration)
#plt.figure()
##plt.plot(myrecording)
#plt.show()
#plt.title('sound')
#sd.play(myrecording, fs)

#dct_rec = fft.dct(myrecording, axis = 0)
#plt.figure()
#plt.plot(dct_rec)
#plt.title('sound dct')
#plt.show()
#%%  Processing stuff
a = 0.95
x = myrecording - a*np.roll(myrecording, -1)
x = x[:, 0]
n = np.arange(N)

w = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
s = x*w
sd.play(s/max(s),fs)

#DCT
S = fft.dct(s, type = 1, axis = 0)
#
P = (1/N)*S**2



fhz_low = 300
fhz_high = 8000
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

#mels = np.zeros(nfilterbank)
#for i in range(nfilterbank):
#    mels[i] = sum(fbank[i, :] * P[:np.shape(fbank)[1]])

#plt.plot(mels)
#melcep = fft.dct(mels, type = 2, axis = 0)[0:12]
#plt.plot(melcep)
#%%
def melceps(samps):
    N = np.shape(samps)[0]
    Nsamps = np.shape(samps)[1]
    n = np.arange(N)
    melcep_out = np.zeros((12,Nsamps))
    samps[np.isnan(samps)] = 0
    for i in range(Nsamps):
        x = samps[:,i] - a*np.roll(samps[:,i], -1)
        #x = x[:, 0]
        w = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
        s = x*w
        #DCT
        S = fft.dct(s, type = 1, axis = 0)
        #power
        P = (1/N)*S**2
        mels = np.zeros(nfilterbank)
        for j in range(nfilterbank):
            mels[j] = sum(fbank[j, :] * P[:np.shape(fbank)[1]])
        melcep_out[:, i] = fft.dct(np.log(mels),type = 1, axis = 0)[0:12]
    return melcep_out

#%% RECORD HIT ME SAMPLES
Nsamps = 20
hitmes = np.zeros((N,Nsamps),dtype = np.float32)
for i in range(Nsamps):
    print(i)
    rec = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
    time.sleep(duration)
    hitmes[:, i] = rec[:,0]
    time.sleep(duration)

#%%
ones = np.zeros((N,Nsamps),dtype = np.float32)
for i in range(Nsamps):
    print(i)
    rec = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
    time.sleep(duration)
    ones[:, i] = rec[:,0]
    time.sleep(duration)

#%%
twos = np.zeros((N,Nsamps),dtype = np.float32)
for i in range(Nsamps):
    print(i)
    rec = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
    time.sleep(duration)
    twos[:, i] = rec[:,0]
    time.sleep(duration)
    
#%%
garbo = np.zeros((N,Nsamps),dtype = np.float32)
for i in range(Nsamps):
    print(i)
    rec = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
    time.sleep(duration)
    garbo[:, i] = rec[:,0]
    time.sleep(duration)
    
#%%
hitme_melcep = melceps(hitmes)
ones_melcep = melceps(ones)
twos_melcep = melceps(twos)
garbo_melcep = melceps(garbo)

#%% PLOT SOME MELCEPS
plt.figure()
plt.plot(hitme_melcep)
plt.figure()
plt.plot(ones_melcep)
plt.figure()
plt.plot(twos_melcep)
plt.figure()
plt.plot(garbo_melcep)

#%% MAKE THE ARRAYS
hitme_dat = np.hstack((np.zeros((Nsamps,1)), hitme_melcep.T))
ones_dat = np.hstack((np.ones((Nsamps,1)), ones_melcep.T))
twos_dat = np.hstack((2*np.ones((Nsamps,1)), twos_melcep.T))
garbo_dat = np.hstack((3*np.ones((Nsamps,1)), garbo_melcep.T))

train_data = np.vstack((hitme_dat, ones_dat, twos_dat, garbo_dat))
np.random.shuffle(train_data)
train_data = train_data.T


#%% QDA BABY
from numpy import linalg

class Predictor_QDA(object):
    """A class that predicts the MPG as either high (1) or low (0) based on other factors about the car.
    
    This class is identical in operation as MPGPredictor_LDA.  But internally, it uses Quadratic
    Discriminant Analysis, rather than Linear.
    
    There are two stages to using this.  First, train the model using data that has known MPG values.
    Then this model can be used to predict the MPG for other data.
    Finally, a third function returns the confusion matrix based on the actual MPG of the test case.
    """
    def __init__(self, predictors, IDs):
        """Initialize given a length-N array of mpg values (either 0 or 1) and a (k x N) array
        of predictors.  
        
        (In this case, k = 7, but your code probably doesn't need to depend on that.)
        """
        self.IDs = IDs
        self.predictors = predictors
        self.train()
        self.N = np.shape(predictors)[1]
        
    def train(self):
        """Set some internal values to use for the LDA predictions based on the training data
        provided by the initializer, self.mpg01 and self.predictors.
        """
        # You should calculate some values here and save them to use later in the predict function.
        #   self.something = ...
        #   self.something_else = ...
        # Basically compute everything you are going to need to do the LDA prediction.

        hitmes = (self.IDs == 0)
        ones = (self.IDs == 1)
        twos = (self.IDs == 2)
        garbo = (self.IDs == 3)
        self.mu_h = np.mean(self.predictors[:,hitmes],axis=1)
        self.mu_o = np.mean(self.predictors[:,ones],axis=1)
        self.mu_t = np.mean(self.predictors[:,twos],axis=1)
        self.mu_g = np.mean(self.predictors[:,garbo],axis=1)
        self.cov_h = np.cov(self.predictors[:,hitmes])
        self.cov_o = np.cov(self.predictors[:,ones])
        self.cov_t = np.cov(self.predictors[:,twos])
        self.cov_g = np.cov(self.predictors[:,garbo])
        self.inv_cov_h = linalg.inv(self.cov_h)
        self.inv_cov_o = linalg.inv(self.cov_o)
        self.inv_cov_t = linalg.inv(self.cov_t)
        self.inv_cov_g = linalg.inv(self.cov_g)
        self.pi_h = float(np.sum(hitmes)) / len(self.IDs)
        self.pi_o = float(np.sum(ones)) / len(self.IDs)
        self.pi_t = float(np.sum(twos)) / len(self.IDs)
        self.pi_g = float(np.sum(garbo)) / len(self.IDs)
        
    def predict(self, predictors):
        """Given a test set of predictors, predict the mpg class (high = 1, low = 0) using Linear
        Discriminant Analysis, based on the values calculated in train()
        
        Returns an array of either 0 or 1 indicating the mpg class.        
        """
        # Equation 4.23:
        # The first term, -0.5 xT Sigma^-1 x, isn't super obvious how to do with
        # numpy math.  If you do x.T.dot(inv_cov).dot(x), this gives you an NxN
        # matrix, of which you only want the diagonal.
        # So diag(x.T.dot(inv_cov).dot(x)) will work.
        #
        # But a handy matrix trick to know is that diag(A.dot(B)) is equal to
        # (A * B.T).sum(axis=1) (Prove this to yourself if it's not obvious.)
        # which is much faster if N is large.  In our case, this yields
        # (x.T.dot(inv_cov) * x.T).sum(axis=1).
        x = predictors
        delta_h = (-0.5 * (x.T.dot(self.inv_cov_h) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_h).dot(self.mu_h) -
                    0.5 * self.mu_h.T.dot(self.inv_cov_h).dot(self.mu_h) -
                    0.5 * np.log(linalg.det(self.cov_h)) +
                    np.log(self.pi_h))
        delta_o = (-0.5 * (x.T.dot(self.inv_cov_o) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_o).dot(self.mu_o) -
                    0.5 * self.mu_o.T.dot(self.inv_cov_o).dot(self.mu_o) -
                    0.5 * np.log(linalg.det(self.cov_o)) +
                    np.log(self.pi_o))
        delta_t = (-0.5 * (x.T.dot(self.inv_cov_t) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_t).dot(self.mu_t) -
                    0.5 * self.mu_t.T.dot(self.inv_cov_t).dot(self.mu_t) -
                    0.5 * np.log(linalg.det(self.cov_t)) +
                    np.log(self.pi_t))
        delta_g = (-0.5 * (x.T.dot(self.inv_cov_g) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_g).dot(self.mu_g) -
                    0.5 * self.mu_g.T.dot(self.inv_cov_g).dot(self.mu_g) -
                   0.5 * np.log(linalg.det(self.cov_g)) +
                   np.log(self.pi_g))
        results = np.zeros(np.shape(predictors)[1])
        for i in range(np.shape(predictors)[1]):
            results[i] = np.argmax([delta_h[i], delta_o[i], delta_t[i], delta_g[i]])
        return results
                

    def calc_accuracy(self, pred, truth):
        """Compute the accuracy for a given set of predicted high/low mpg values given
        """
        hitmes = truth == 0
        ones = truth == 1
        twos = truth == 2
        garbo = truth == 3
        print('Correct hit mes, Missed hit mes:'+ str(sum(hitmes[pred==0])) + ', ' + str(sum(hitmes)-sum(hitmes[pred==0])))
        print('Correct ones, Missed ones:'+ str(sum(ones[pred==1])) + ', ' + str(sum(ones)-sum(ones[pred==1])))
        print('Correct twos, Missed twos:'+ str(sum(twos[pred==2])) + ', ' +  str(sum(twos)-sum(twos[pred==2])))
        print('Correct garbo, Missed garbo:'+ str(sum(garbo[pred==3])) + ', ' + str(sum(garbo)-sum(garbo[pred==3])))
     
#%% TEST
IDS = train_data[0, :]
predictors = train_data[1:, :]
ntrain = int(np.shape(train_data)[1]*3/4)
command_predictor = Predictor_QDA(predictors, IDS)

# Predicting the training data should work very well.
pred_training = command_predictor.predict(predictors[:,:ntrain])
print('On training data, the results are:')
command_predictor.calc_accuracy(pred_training, IDS[:ntrain])
print()

# Now try it out on the test data.
pred_val = command_predictor.predict(predictors[:,ntrain:])
print('On test data, the results are:')
command_predictor.calc_accuracy(pred_val, IDS[ntrain:])

#%%
map = {0:"Hit me", 1:"One", 2:"Two", 3: "Garbo"}
def voice_recog():
    print("Say command")
    duration = 1 #seconds
    fs = 16000
    N = duration*fs
    samps = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
    time.sleep(duration*3)
    N = np.shape(samps)[0]
    Nsamps = np.shape(samps)[1]
    n = np.arange(N)
    melcep_out = np.zeros((12,Nsamps))
    samps[np.isnan(samps)] = 0
    for i in range(Nsamps):
        x = samps[:,i] - a*np.roll(samps[:,i], -1)
        #x = x[:, 0]
        w = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
        s = x*w
        #DCT
        S = fft.dct(s, type = 1, axis = 0)
        #power
        P = (1/N)*S**2
        mels = np.zeros(nfilterbank)
        for j in range(nfilterbank):
            mels[j] = sum(fbank[j, :] * P[:np.shape(fbank)[1]])
        melcep_out[:, i] = fft.dct(np.log(mels),type = 1, axis = 0)[0:12]
        
    prediction = mpg_predictor.predict(melcep_out)
    print(map[prediction[0]])
#%%  
voice_recog()

