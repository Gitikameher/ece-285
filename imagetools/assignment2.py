""" Assignment 2

COMPLETE THIS FILE

Your name here:

"""

from .provided import *
from math import e

def kernel(name, tau=1, eps=1e-3):
    if name is 'gaussian':
        i=0
        while e**(- (i**2)/(2*(tau**2)))> eps:
            i+=1
        s1=i-1;s2=i-1
        x = np.arange(0-s1, 2*s1+1-s1, 1)
        y = np.arange(0-s2, 2*s2+1-s2, 1)
        xx, yy = np.meshgrid(x, y)
        nu=gaussian_kernel(xx,yy,tau)  
        nu= nu/sum(sum(nu))
        
    elif name is 'exponential':
        i=0
        while e**(- i/tau)>eps:
            i+=1
        s1=i-1;s2=i-1
        x = np.arange(0-s1, 2*s1+1-s1, 1)
        y = np.arange(0-s2, 2*s2+1-s2, 1)
        xx, yy = np.meshgrid(x, y)
        nu=exponential_kernel(xx,yy,tau)  
        nu= nu/sum(sum(nu))
    elif name is 'box':
        s1=tau;s2=tau
        nu=np.ones((2*s1+1, 2*s2+1))/((2*s1+1) * (2*s2+1))
    return nu

def gaussian_kernel(i,j, tau):
    return e**(- (i**2 + j**2)/(2*(tau**2)))

def exponential_kernel(i, j, tau):
    return e**(-(i**2 + j**2)**0.5/tau)

def convolve_naive(x, nu):
    n1, n2 = x.shape[:2]
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    xconv = np.zeros(x.shape)
    for i in range(s1, n1-s1):
        for j in range(s2, n2-s2):
            count=np.zeros(x.shape[2]) 
            for k in range(-s1, s1+1):
                for l in range(-s2, s2+1):
                    count+=x[i-k,j-l,:]*nu[k+s1, l+s2]
            xconv[i,j,:]=count
    return xconv

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

def convolve(x, nu, boundary='periodical'):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv+=shift(x, k, l, boundary=boundary)*nu[k+s1,l+s2]
    return xconv
            
            