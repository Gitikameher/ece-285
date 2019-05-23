""" Assignment 5

COMPLETE THIS FILE

Your name here:

"""
import numpy as np
from .assignment4 import *

def kernel2fft(nu, n1, n2, separable=None):
    if separable==None:
        s1=int((nu.shape[0]-1)/2);s2=int((nu.shape[1]-1)/2)
    else:
        s1=int((nu[0].shape[0]-1)/2);s2=int((nu[1].shape[1]-1)/2)
    if separable=='product':
        nu=np.outer(nu[0],nu[1])
    if separable=='sum':
        tmp=np.zeros((2*s1+1, 2*s2+1))
        tmp[:, s2]= nu[0][:,0]
        tmp[s1,:]+=nu[1][0,:]
        nu=tmp
    tmp=np.zeros((n1, n2))
    tmp[:s1+1, :s2+1] = nu[s1:(2*s1+1), s2:(2*s2+1)]
    if s2>0:
        tmp[:s1+1, -s2:] = nu[s1:2*s1+1, :s2] 
    if s1>0:
        tmp[-s1:, :s2+1] = nu[:s1, s2:2*s2+1]
    if s1>0 and s2>0 :    
        tmp[-s1:, -s2:] = nu[:s1, :s2]
    return np.fft.fft2(tmp)

def convolvefft(x, lbd):
    lbda=lbd.reshape(*lbd.shape, *[1]*(x.ndim-lbd.ndim))
    return np.real(np.fft.ifft2(np.fft.fft2(x,axes=(0, 1))*lbda, axes=(0, 1)))

def kernel(name, tau=1, eps=1e-3, s1=None, s2=None):
    if name=='motion':
        nu=np.load('assets/motionblur.npy')
    if name.startswith('gaussian'): 
        if s1==None and s2==None:
            i=0
            while e**(- (i**2)/(2*(tau**2)))> eps:
                i+=1
            s1=i-1;s2=i-1
        if name.endswith('1'):
            s2=0
        elif name.endswith('2'):
            s1=0
        x = np.arange(0-s1, 2*s1+1-s1, 1)
        y = np.arange(0-s2, 2*s2+1-s2, 1)
        xx, yy = np.meshgrid(x, y)
        nu=gaussian_kernel(xx,yy,tau)  
        nu= nu/sum(sum(nu))
        
    elif name.startswith('exponential'):
        if tau==0:
            s1=0;s2=0
        else:
            i=0
            while e**(- i/tau)>eps:
                i+=1
            s1=i-1;s2=i-1
        if name.endswith('1'):
            s2=0
        elif name.endswith('2'):
            s1=0
        x = np.arange(0-s1, 2*s1+1-s1, 1)
        y = np.arange(0-s2, 2*s2+1-s2, 1)
        xx, yy = np.meshgrid(x, y)
        nu=exponential_kernel(xx,yy,tau)  
        nu= nu/sum(sum(nu))
        
    elif name.startswith('box'):
        s1=abs(math.floor(tau));s2=abs(math.floor(tau))
        nu=np.ones((2*s1+1, 2*s2+1))/((2*s1+1) * (2*s2+1))
        
    elif name is 'grad1_forward':
        nu = np.array([[0],[-1],[1]])
    elif name is 'grad1_backward':
        nu = np.array([[-1],[1],[0]])
    elif name is 'grad2_forward':
        nu = np.array([[0,-1,1]])
    elif name is 'grad2_backward':
        nu = np.array([[-1,1,0]])
    elif name is 'laplacian2_':
        nu = np.array([[1,-2, 1]])
    elif name is 'laplacian1_':
        nu = np.array([[1],[-2],[1]]) 
    if name.endswith('1'):
        return nu[0].reshape((2*s1+1,1))
    elif name.endswith('2'):
        return nu[:,0].reshape((1,2*s2+1))
    else:
        return nu
    