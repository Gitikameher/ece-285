""" Assignment 6

COMPLETE THIS FILE

Your name here:

"""
import numpy as np
from .assignment5 import *
from math import e
from .provided import *

def average_power_spectral_density(x):
    avg=np.zeros((x[0].shape[0], x[0].shape[1]))
    for k in range(len(x)):
        k1=np.mean(np.fft.fft2(x[k], axes=(0,1)), axis=-1)
        avg+=(np.absolute(k1))**2
    return avg/len(x)

def deconvolve_naive(y, lbd, return_transfer=False):
    hhat= 1/lbd
    xdec= convolvefft(y, hhat)
    if return_transfer:
        return xdec, hhat
    else:
        return xdec
    
def mean_power_spectrum_density(apsd):
    n1=apsd.shape[0];n2=apsd.shape[1]
    s_avg= np.log(apsd) - np.log(n1) - np.log(n2)
    u,v= fftgrid(n1, n2)
    w= np.sqrt( (u/n1)**2 + (v/n2)**2);w[0,0]=1
    t=np.log(w);N= (n1*n2)-1
    alpha= (N*(s_avg*t).sum()- (s_avg.sum())*(t.sum()))/(N*(t*t).sum() - (t.sum())**2)
    beta= (s_avg.sum() - alpha*(t.sum()))/N 
    mpsd= n1*n2*(e**beta)*(w**alpha)
    mpsd[0,0]=np.inf
    return mpsd, alpha, beta

def deconvolve_wiener(x, lbd, sig, mpsd, return_transfer=False):
    n1=mpsd.shape[0];n2=mpsd.shape[1]
    num=1;temp= (n1*n2*(sig)**2)/mpsd
    temp1=(np.absolute(lbd))**2
    den= 1+ (temp/temp1)
    hhat=(num/den)/lbd
    xdec=convolvefft(x, hhat)
    return xdec,hhat 
    
