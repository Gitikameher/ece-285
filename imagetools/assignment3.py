""" Assignment 3

COMPLETE THIS FILE

Your name here:

"""

from .assignment2 import *
import math
from math import e

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

def gaussian_kernel(i,j, tau):
    return e**(- (i**2 + j**2)/(2*(tau**2)))

def exponential_kernel(i, j, tau):
    return e**(-(i**2 + j**2)**0.5/tau)

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

def convolve(x, nu, boundary='periodical', separable=None):
    if separable==None:
        xconv = np.zeros(x.shape)
        s1 = int((nu.shape[0] - 1) / 2)
        s2 = int((nu.shape[1] - 1) / 2)
        for k in range(-s1, s1+1):
            for l in range(-s2, s2+1):
                xconv+=shift(x, -k, -l, boundary=boundary)*nu[k+s1,l+s2]
        
    elif separable=='product':
        s1 = int((nu[0].shape[0] - 1) / 2)
        s2 = int((nu[1].shape[1] - 1) / 2)
        temp=np.zeros(x.shape)
        for k in range(-s1, s1+1):
            temp+=shift(x, -k, 0, boundary=boundary)*nu[0][k+s1,0]
        for l in range(-s2, s2+1):
            xconv+=shift(temp, 0, -l, boundary=boundary)*nu[1][0,l+s2]
            
    elif separable=='sum':
        s1 = int((nu[0].shape[0] - 1) / 2)
        s2 = int((nu[1].shape[1] - 1) / 2)
        for k in range(-s1, s1+1):
            xconv+=shift(x, -k, 0, boundary=boundary)*nu[0][k+s1,0]
        for l in range(-s2, s2+1):
            xconv+=shift(x, 0, -l, boundary=boundary)*nu[1][0,l+s2]
    return xconv         
        
def laplacian(x, boundary='periodical'):
    nu1=kernel('laplacian1_')
    nu2=kernel('laplacian2_')
    nu=(nu1,nu2)
    return convolve(x, nu, boundary, separable='sum')

def grad(x, boundary='periodical'):
    nu1=kernel('grad1_forward')
    nu2=kernel('grad2_forward')
    x1=convolve(x, nu1, boundary, separable=None)
    x2=convolve(x, nu2, boundary, separable=None)
    return np.stack((x1, x2), axis=2)

def div(f, boundary='periodical'):
    nu1=kernel('grad1_backward')
    nu2=kernel('grad2_backward')
    x1=convolve(f[:,:,0], nu1, boundary, separable=None)
    x2=convolve(f[:,:,1], nu2, boundary, separable=None)
    return x1+x2