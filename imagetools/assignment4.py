""" Assignment 4

COMPLETE THIS FILE

Your name here:

"""

from .assignment3 import *
import math
from math import e
import numpy as np

def shift(x, k, l, boundary='periodical'):
    n1, n2 = x.shape[:2]
    
    if boundary is 'periodical':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        
    elif boundary is 'extension':
        if k>=0:
            irange=list(range(k, n1))+ [n1-1]*k
        else:
            irange=[0]*abs(k+1)+list(range(n1+k+1)) 
        if l>=0:
            jrange=list(range(l, n2))+ [n2-1]*l
        else:
            jrange=[0]*abs(l+1)+list(range(n2+l+1)) 
        xshifted = x[irange, :][:, jrange]
        
    elif boundary is 'zero_padding':
        irange = np.mod(np.arange(n1) + k, n1)
        jrange = np.mod(np.arange(n2) + l, n2)
        xshifted = x[irange, :][:, jrange]
        if k>=0:
            xshifted[n1-k:, :,:]=0
        else:
            xshifted[:-k, :,:]=0
        if l>=0:
            xshifted[:, n2-l:,:]=0
        else:
            xshifted[:, :-l,:]=0
            
    elif boundary is 'mirror':
        if k>=0:
            irange=list(range(k, n1))+ list(reversed(list(range(n1-k,n1))))
        else:
            irange=list(reversed(list(range(-k-1))))+list(range(n1+k+1)) 
        if l>=0:
            jrange=list(range(l, n2))+ list(reversed(list(range(n2-l,n2))))
        else:
            jrange=list(reversed(list(range(-l-1))))+list(range(n2+l+1)) 
        xshifted = x[irange, :][:, jrange]
    return xshifted

def bilateral_naive(y, sig, s1=2, s2=2, h=1):
    n1, n2 = y.shape[:2]
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            temp=0;temp1=0
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    dist2 = ((y[i + k, j + l] - y[i, j])**2).mean()
                    temp2=kernel_function(h, dist2, sig, c)
                    temp+=temp2*(y[i + k, j + l])
                    temp1+=temp2       
            x[i,j]=temp
            Z[i,j]=temp1
    Z[Z == 0] = 1
    x = x / Z
    return x

def kernel_function(h, alpha, sig, c):
    y=alpha-(2*h*(sig**2))
    x=2**(1.5)*h*(sig**2)/(c**0.5)
    return e**(-np.maximum(y,0)/x)

def kernel_function_nl(h, alpha, sig, c,p):
    y=alpha-(2*h*(sig**2))
    x=2**(1.5)*h*(sig**2)/((c*p)**0.5)
    return e**(-np.maximum(y,0)/x) 

def nlmeans_naive(y, sig, s1=2, s2=2, p1=1, p2=1, h=1):
    n1, n2 = y.shape[:2];P=(2*p1+1)*(2*p2+1)
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    for i in range(s1, n1-s1-p1):
        for j in range(s2, n2-s2-p2):
            temp=0;temp1=0
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    dist2 = 0
                    for u in range(-p1, p1 + 1):
                        for v in range(-p2, p2 + 1):
                            dist2 += ((y[i + k+p1, j + l+p2] - y[i+p1, j+p2])**2).mean()
                    temp2=kernel_function_nl(h, dist2/P, sig, c, P)
                    temp+=temp2*(y[i + k, j + l])
                    temp1+=temp2
            x[i,j]=temp
            Z[i,j]=temp1
    Z[Z == 0] = 1
    x = x / Z
    return x

def bilateral(y, sig, s1=10, s2=10, h=1, boundary='mirror'):
    n1, n2 = y.shape[:2];c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            t=shift(y, k, l, boundary=boundary)
            dist2=(np.sum((t-y)**2,axis=2)).reshape((y.shape[0],y.shape[1],1))
            temp=kernel_function(h, dist2/c, sig, c)
            x+=temp*t
            Z+=temp
    return x/Z

def nlmeans(y, sig, s1=7, s2=7, p1=None, p2=None, h=1, boundary='periodical'):
    p1 = (1 if y.ndim == 3 else 2) if p1 is None else p1
    p2 = (1 if y.ndim == 3 else 2) if p2 is None else p2
    n1, n2 = y.shape[:2]
    p=(2*p1+1)*(2*p2+1)
    c = y.shape[2] if y.ndim == 3 else 1
    x = np.zeros(y.shape)
    Z = np.zeros((n1, n2, *[1] * (y.ndim - 2)))
    nu=kernel('box',p1)
    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            temp1=shift(y, k, l,boundary=boundary)
            dist2=(np.sum((temp1-y)**2,axis=2)).reshape((y.shape[0],y.shape[1],1))
            dist2=convolve(dist2, nu, boundary=boundary, separable=None)
            temp2=kernel_function_nl(h, dist2/c, sig, c, p)
            x+=temp2*temp1
            Z+=temp2
    return x/Z


def psnr(x, x0):
    if x.max()<=1:
        R=1
    elif x.max()<=255:
        R=255
    den=np.mean((x-x0)**2)
    return 10*math.log((R**2)/den,10)
    
def kernel(name, tau=1, eps=1e-3):
    if name.startswith('gaussian'):            
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
        nu=np.ones((2*s1+1, 2*s2+1))/((2*s1+1)*(2*s2+1))
        
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

def convolve(x, nu, boundary='periodical', separable=None):
    if separable==None:
        xconv = np.zeros(x.shape)
        s1 = int((nu.shape[0] - 1) / 2)
        s2 = int((nu.shape[1] - 1) / 2)
        for k in range(-s1, s1+1):
            for l in range(-s2, s2+1):
                xconv+=shift(x, -k, -l, boundary=boundary)*nu[k+s1,l+s2]
        
    elif separable=='product':
        x1=convolve(x, nu[0], boundary=boundary)
        xconv=convolve(x1, nu[1], boundary=boundary)
        
    elif separable=='sum':
        x1=convolve(x, nu[0], boundary=boundary)
        x2=convolve(x, nu[1], boundary=boundary)
        xconv=x1+x2
    return xconv 