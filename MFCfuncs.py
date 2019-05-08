"""
Nibl

@author: Bhaskar Abhiraman and Jason Kaufmann
"""

import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import time
import numpy as np
from numpy import linalg

import rcpy
import rcpy.gpio as gpio
import rcpy.adc as adc

from micadc import record
import os
from PIL import Image


def M(f):
    return 1125.0*np.log(1.0+f/700.0)

def Mi(m):
    return 700.0*(np.exp(m/1125.0)-1.0)


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

#mels = np.zeros(nfilterbank)
#for i in range(nfilterbank):
#    mels[i] = sum(fbank[i, :] * P[:np.shape(fbank)[1]])

#plt.plot(mels)
#melcep = fft.dct(mels, type = 2, axis = 0)[0:12]
#plt.plot(melcep)
#%%
def melceps(samps):
    a = 0.95
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
        #S = fft.dct(s, type = 1, axis = 0)
        S = np.real(np.fft.rfft(s, axis = 0))
        #print('S shape:', np.shape(S))
        #print('St shape:', np.shape(St))
        #power
        P = (1/N)*S**2
        mels = np.zeros(nfilterbank)
        for j in range(nfilterbank):
            mels[j] = sum(fbank[j, :] * P[:np.shape(fbank)[1]])
        melcep_out[:, i] = np.real(np.fft.rfft(np.log(mels),axis = 0)[0:12])
    return melcep_out

#%% RECORD SAMPLES
def rec_hitmes(Nsamp, duration, fs, ch):
    N = fs*duration
    hitmes = np.zeros((N,Nsamp),dtype = np.float32)
    for i in range(Nsamp):
        print('Say HIT ME: ' + str(i+1) + ' / ' + str(Nsamp))
        rec = (record(duration, fs, ch) - 2048)/2048
        hitmes[:, i] = rec[:,0]
        time.sleep(duration)
    
    hitmes = np.asarray(hitmes)
    hitmes[np.isnan(hitmes)] = 0  
    hitmes[abs(hitmes)>1] = 0   
    hitme_melcep = melceps(hitmes)
    np.save('hitme_melcep.npy', hitme_melcep)
    
def rec_lefts(Nsamp, duration, fs, ch):
    N = fs*duration
    lefts = np.zeros((N,Nsamp),dtype = np.float32)
    for i in range(Nsamp):
        print('Say LEFT: ' + str(i+1) + ' / ' + str(Nsamp))
        rec = (record(duration, fs, ch) - 2048)/2048
        lefts[:, i] = rec[:,0]
        time.sleep(duration)
    
    lefts = np.asarray(lefts)
    lefts[np.isnan(lefts)] = 0  
    lefts[abs(lefts)>1] = 0   
    left_melcep = melceps(lefts)
    np.save('left_melcep.npy', left_melcep)
    
def rec_rights(Nsamp, duration, fs, ch):
    N = fs*duration
    rights = np.zeros((N,Nsamp),dtype = np.float32)
    for i in range(Nsamp):
        print('Say RIGHT: ' + str(i+1) + ' / ' + str(Nsamp))
        rec = (record(duration, fs, ch) - 2048)/2048
        rights[:, i] = rec[:,0]
        time.sleep(duration)
    
    rights = np.asarray(rights)
    rights[np.isnan(rights)] = 0  
    rights[abs(rights)>1] = 0   
    right_melcep = melceps(rights)
    np.save('right_melcep.npy', right_melcep)
    
def rec_ups(Nsamp, duration, fs, ch):
    N = fs*duration
    ups = np.zeros((N,Nsamp),dtype = np.float32)
    for i in range(Nsamp):
        print('Say UP: ' + str(i+1) + ' / ' + str(Nsamp))
        rec = (record(duration, fs, ch) - 2048)/2048
        ups[:, i] = rec[:,0]
        time.sleep(duration)
    
    ups = np.asarray(ups)
    ups[np.isnan(ups)] = 0  
    ups[abs(ups)>1] = 0   
    up_melcep = melceps(ups)
    np.save('up_melcep.npy', up_melcep)

def rec_downs(Nsamp, duration, fs, ch):
    N = fs*duration
    downs = np.zeros((N,Nsamp),dtype = np.float32)
    for i in range(Nsamp):
        print('Say DOWN: ' + str(i+1) + ' / ' + str(Nsamp))
        rec = (record(duration, fs, ch) - 2048)/2048
        downs[:, i] = rec[:,0]
        time.sleep(duration)
    
    downs = np.asarray(downs)
    downs[np.isnan(downs)] = 0  
    downs[abs(downs)>1] = 0   
    down_melcep = melceps(downs)
    np.save('down_melcep.npy', down_melcep)

    
def rec_garbo(Nsamp, duration, fs, ch):
    N = fs*duration
    garbo = np.zeros((N,Nsamp),dtype = np.float32)
    for i in range(Nsamp):
        print('Say NOTHING: ' + str(i+1) + ' / ' + str(Nsamp))
        rec = (record(duration, fs, ch) - 2048)/2048
        garbo[:, i] = rec[:,0]
        time.sleep(duration)
    
    garbo = np.asarray(garbo)
    garbo[np.isnan(garbo)] = 0  
    garbo[abs(garbo)>1] = 0   
    garbo_melcep = melceps(garbo)
    np.save('garbo_melcep.npy', garbo_melcep)

#%% QDA BABY
class Predictor_QDA(object):
    """Predicts class of vocal command
    """
    def __init__(self, predictors, IDs):
        """
        """
        self.IDs = IDs
        self.predictors = predictors
        self.train()
        self.N = np.shape(predictors)[1]
        
    def train(self):
        """Preemptively calculate mean and covariance matrices
        """
        # You should calculate some values here and save them to use later in the predict function.
        #   self.something = ...
        #   self.something_else = ...
        # Basically compute everything you are going to need to do the LDA prediction.

        hitmes = (self.IDs == 0)
        lefts = (self.IDs == 1)
        rights = (self.IDs == 2)
        ups = (self.IDs == 3)
        downs = (self.IDs == 4)
        garbo = (self.IDs == 5)
        self.mu_h = np.mean(self.predictors[:,hitmes],axis=1)
        self.mu_l = np.mean(self.predictors[:,lefts],axis=1)
        self.mu_r = np.mean(self.predictors[:,rights],axis=1)
        self.mu_u = np.mean(self.predictors[:,ups],axis=1)
        self.mu_d = np.mean(self.predictors[:,downs],axis=1)
        self.mu_g = np.mean(self.predictors[:,garbo],axis=1)
        
        self.cov_h = np.cov(self.predictors[:,hitmes])
        self.cov_l = np.cov(self.predictors[:,lefts])
        self.cov_r = np.cov(self.predictors[:,rights])
        self.cov_u = np.cov(self.predictors[:,ups])
        self.cov_d = np.cov(self.predictors[:,downs])
        self.cov_g = np.cov(self.predictors[:,garbo])
        
        self.inv_cov_h = linalg.inv(self.cov_h)
        self.inv_cov_l = linalg.inv(self.cov_l)
        self.inv_cov_r = linalg.inv(self.cov_r)
        self.inv_cov_u = linalg.inv(self.cov_u)
        self.inv_cov_d = linalg.inv(self.cov_d)
        self.inv_cov_g = linalg.inv(self.cov_g)
        
        self.pi_h = float(np.sum(hitmes)) / len(self.IDs)
        self.pi_l = float(np.sum(lefts)) / len(self.IDs)
        self.pi_r = float(np.sum(rights)) / len(self.IDs)
        self.pi_u = float(np.sum(ups)) / len(self.IDs)
        self.pi_d = float(np.sum(downs)) / len(self.IDs)
        self.pi_g = float(np.sum(garbo)) / len(self.IDs)
        
    def predict(self, predictors):
        """Predict output class with QDA   
        """
        x = predictors
        delta_h = (-0.5 * (x.T.dot(self.inv_cov_h) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_h).dot(self.mu_h) -
                    0.5 * self.mu_h.T.dot(self.inv_cov_h).dot(self.mu_h) -
                    0.5 * np.log(linalg.det(self.cov_h)) +
                    np.log(self.pi_h))
        delta_l = (-0.5 * (x.T.dot(self.inv_cov_l) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_l).dot(self.mu_l) -
                    0.5 * self.mu_l.T.dot(self.inv_cov_l).dot(self.mu_l) -
                    0.5 * np.log(linalg.det(self.cov_l)) +
                    np.log(self.pi_l))
        delta_r = (-0.5 * (x.T.dot(self.inv_cov_r) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_r).dot(self.mu_r) -
                    0.5 * self.mu_r.T.dot(self.inv_cov_r).dot(self.mu_r) -
                    0.5 * np.log(linalg.det(self.cov_r)) +
                    np.log(self.pi_r))
        delta_u = (-0.5 * (x.T.dot(self.inv_cov_u) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_u).dot(self.mu_u) -
                    0.5 * self.mu_u.T.dot(self.inv_cov_u).dot(self.mu_u) -
                    0.5 * np.log(linalg.det(self.cov_u)) +
                    np.log(self.pi_u))
        delta_d = (-0.5 * (x.T.dot(self.inv_cov_d) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_d).dot(self.mu_d) -
                    0.5 * self.mu_d.T.dot(self.inv_cov_d).dot(self.mu_d) -
                    0.5 * np.log(linalg.det(self.cov_d)) +
                    np.log(self.pi_d))
        delta_g = (-0.5 * (x.T.dot(self.inv_cov_g) * x.T).sum(axis=1) +
                    x.T.dot(self.inv_cov_g).dot(self.mu_g) -
                    0.5 * self.mu_g.T.dot(self.inv_cov_g).dot(self.mu_g) -
                   0.5 * np.log(linalg.det(self.cov_g)) +
                   np.log(self.pi_g))
        
        results = np.zeros(np.shape(predictors)[1])
        for i in range(np.shape(predictors)[1]):
            results[i] = np.argmax([delta_h[i], delta_l[i], delta_r[i], delta_u[i], delta_d[i], delta_g[i]])
        return results
                

    def calc_accuracy(self, pred, truth):
        """Compute the accuracy for a test set of samples
        """
        hitmes = truth == 0
        lefts = truth == 1
        rights = truth == 2
        ups = truth == 3
        downs = truth == 4
        garbo = truth == 5
        print('Correct hit mes, Missed hit mes:'+ str(sum(hitmes[pred==0])) + ', ' + str(sum(hitmes)-sum(hitmes[pred==0])))
        print('Correct lefts, Missed lefts:'+ str(sum(lefts[pred==1])) + ', ' + str(sum(lefts)-sum(lefts[pred==1])))
        print('Correct rights, Missed rights:'+ str(sum(rights[pred==2])) + ', ' +  str(sum(rights)-sum(rights[pred==2])))
        print('Correct ups, Missed ups:'+ str(sum(ups[pred==3])) + ', ' +  str(sum(ups)-sum(ups[pred==3])))
        print('Correct downs, Missed downs:'+ str(sum(downs[pred==4])) + ', ' +  str(sum(downs)-sum(downs[pred==4])))
        print('Correct garbo, Missed garbo:'+ str(sum(garbo[pred==5])) + ', ' + str(sum(garbo)-sum(garbo[pred==5])))
     
#%% TEST
def train(frac, melceps, Nsamps, Nsampsgarbo):
    hitme_melcep, left_melcep, right_melcep, up_melcep, down_melcep, garbo_melcep = melceps
    
    hitme_dat = np.hstack((np.zeros((Nsamps,1)), hitme_melcep.T))
    left_dat = np.hstack((1*np.ones((Nsamps,1)), left_melcep.T))
    right_dat = np.hstack((2*np.ones((Nsamps,1)), right_melcep.T))
    up_dat = np.hstack((3*np.ones((Nsamps,1)), up_melcep.T))
    down_dat = np.hstack((4*np.ones((Nsamps,1)), down_melcep.T))
    garbo_dat = np.hstack((5*np.ones((Nsampsgarbo,1)), garbo_melcep.T))
    
    train_data = np.vstack((hitme_dat[:int(frac*np.shape(hitme_dat)[0])], left_dat[:int(frac*np.shape(left_dat)[0])], right_dat[:int(frac*np.shape(right_dat)[0])], up_dat[:int(frac*np.shape(up_dat)[0])], down_dat[:int(frac*np.shape(down_dat)[0])], garbo_dat[:int(frac*np.shape(garbo_dat)[0])],
                                      hitme_dat[int(frac*np.shape(hitme_dat)[0]):], left_dat[int(frac*np.shape(left_dat)[0]):], right_dat[int(frac*np.shape(right_dat)[0]):], up_dat[int(frac*np.shape(up_dat)[0]):], down_dat[int(frac*np.shape(down_dat)[0]):], garbo_dat[int(frac*np.shape(garbo_dat)[0]):]))
    train_data = train_data.T          
    #print('shape of train_data', np.shape(train_data))
    t0 = time.time()
    IDS = train_data[0, :]
    #print(IDS)
    predictors = train_data[1:, :]
    ntrain = int(np.shape(train_data)[1]*frac)
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
    t1 = time.time()
    print("time it took:" + str(t1-t0))
    
    return command_predictor

def comm(a, b, c):
    file = open("data.txt","w") 
    file.writelines([str(a) + '\n', str(b)+ '\n', str(c)])
    file.close()
    command = "/opt/source/librobotcontrol/examples/bin/comm"
    os.system(command)


#%% Continuous voice recog
def cont_voice_recog(command_predictor):
    outputSignal = gpio.Output(3, 2)
    inputSignal = gpio.Input(3, 1)
    outputSignal.set(gpio.LOW)
    map = {0:"Hit me", 1:"Left", 2:"Right", 3:"Up", 4:"Down", 5: "Garbo"}
    #print("Say command")
    window = 1 #seconds
    steps = 1
    shift = window/steps
    fs = 6000
    Nshift = int(shift*fs)
    N = window*fs
    n = np.arange(N)
    w = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
    a = 0.95
    samps = np.zeros((int(fs*window),1))
    while True:
        
        newsamps = (record(shift, fs, ch) - 2048)/2048
        t1 = time.time()
        newsamps[np.isnan(newsamps)] = 0
        newsamps[abs(newsamps)>1] = 0     
        samps = np.append(samps[Nshift:], newsamps)
        if np.mean(samps**2) > .001:
            melcep_out = np.zeros((12,1))
            x = samps - a*np.roll(samps, -1)
            #x = x[:, 0]
            s = x*w
            #DCT
            S = np.real(np.fft.rfft(s, axis = 0))
            #power
            P = (1/N)*S**2
            mels = np.zeros(nfilterbank)
            for j in range(nfilterbank):
                mels[j] = sum(fbank[j, :] * P[:np.shape(fbank)[1]])
            melcep_out[:, 0] = np.real(np.fft.rfft(np.log(mels), axis = 0)[0:12])
            prediction = command_predictor.predict(melcep_out)
            #if prediction[0] != 3:
            print(map[prediction[0]])
            t2 = time.time()
            print('delay: ', t2-t1)
            if prediction[0] == 0:
                comm(-1,-1,-1)
                time.sleep(1)
                scan()
            elif prediction[0] == 2:
                comm(-2,-2,-2)
                time.sleep(1)
                track()
                #break
    #comm(-1, -1, -1)
    #time.sleep(1)
    #scan()
                #take_pic()
def take_pic(i):
    
    try:
        photoNum = i #photoNum = int(photoArr[0])
        command = "fswebcam -r 640x480 --no-banner scan" + str(photoNum) +".jpg"
        os.system(command)
        print( "scan" + str(photoNum) +".jpg created")

    except:
        print("Error: scan" + str(photoNum) + " not saved")

    return photoNum

def processing(im):
    (h0, w0) = im.size
    fact = 35 #scaling factor

    t1 = time.time()
    im = im.resize((int(h0 / fact), int(w0 / fact)))
    array = np.array(im.getdata())
    (h, w) = im.size

    ## Ratios of R/G, R/B, G/B
    cosrgbmax = [0.975, 2.31, 3.07]
    cosrgbmin = [0.67, 1.43, 1.72]
    wanrgbmax = [19.62, 2.04, 0.246]
    wanrgbmin = [6.46, 1.4, 0.087]

    num = 0
    loch= 0
    locw= 0

    for i in range(w * h):
        if (np.random.random() > 0):
            r, g, b = array[i]
            ratios = [0,0,0]
            if g is not 0:
                ratios[0] = r/g
            else:
                ratios[0] = 255
            if b is not 0:
                ratios[1] = r/b
                ratios[2] = g/b
            else:
                ratios[1] = 255
                ratios[2] = 255
            if sum(np.greater_equal(ratios, cosrgbmin)) == 3 and sum(np.less_equal(ratios, cosrgbmax)) == 3:
                hi = i % h
                wi = int(i / h)
                #justcoswan[hi, wi] = 1
                num += 1
                loch += hi
                locw += wi
            elif sum(np.greater_equal(ratios, wanrgbmin)) == 3 and sum(np.less_equal(ratios, wanrgbmax)) == 3:
                hi = i % h
                wi = int(i / h)
                #justcoswan[hi, wi] = 1
                num += 1
                loch += hi
                locw += wi
    if num is not 0:

        loch = int(100*loch/(num*h))
        locw = int(100*locw/(num*w))
    else:
        loch = 101
        locw = 101
    
    if num > 128:
        num = 128

    t2 = time.time()
    #print('Processing took:', t2 - t1)
    return loch, locw, num

def send_email(img_file):
    subject = "FROM THE GOD DAMN NIBL"
    body = "TEST"
    sender_email = "niblese350@gmail.com"
    receiver_email = "5619720038@mms.att.net"
    password = "welovechris"
    
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails
    
    # Add body to email
    message.attach(MIMEText(body, "plain"))
    
    filename = img_file  # In same directory as script
    
    # Open file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    
    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)
    s = "attachment; filename= "+filename  
    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        s,
    )
    
    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()
    
    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

def scan():
    inputSignal = gpio.Input(3,1)
    outputSignal = gpio.Output(3,2)
    for i in range(8):
        num = take_pic(i)
        im = Image.open("scan"+str(i)+".jpg")
        #print(im.format, im.size, im.mode)
        h, w, mass = processing(im)
        #comm(mass, h, w)
        outputSignal.set(gpio.HIGH)
        time.sleep(0.01)
        outputSignal.set(gpio.LOW)
        comm(mass, h, w)
        while(not inputSignal.is_high()):
            pass
    while(not inputSignal.is_high()):
        pass
    n = take_pic(9)
    #send_email('scan9.jpg')
    
def track():
    inputSignal = gpio.Input(3,1)
    outputSignal = gpio.Output(3,2)
    i = 0
    while (1):
        num = take_pic(i)
        im = Image.open("scan"+str(i)+".jpg")
        h,w,mass = processing(im)
        outputSignal.set(gpio.HIGH)
        time.sleep(0.1)
        outputSignal.set(gpio.LOW)
        comm(mass,h,w)
        while(not inputSignal.is_high()):
            pass
        i+=1
        i%=10

    
