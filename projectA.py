""" Project A

COMPLETE THIS FILE

Your names here:

"""
import numpy as np
from .assignment6 import *

class Identity(LinearOperator):
    def __init__(self, ishape):
        self.shape=ishape
        oshape=ishape
        LinearOperator.__init__(self, ishape, oshape)
        self.H= np.ones(self.ishape, dtype=int)
                
    def __call__(self, x):
        return convolvefft(x, self.H)
    
    def adjoint(self, x):
        return convolvefft(x, np.conjugate(self.H))
    
    def gram(self, x):
        return convolvefft(x, np.conjugate(self.H)*self.H)
    
    def gram_resolvent(self, x, tau):
        return cg(lambda z: z + tau * self.gram(z), x)
        
class Convolution(LinearOperator):
    def __init__(self, shape, nu, separable=None):
        self.n1=shape[0];self.n2=shape[1]
        self.nu=nu
        self.separable=separable
        self.ker=kernel2fft(self.nu, self.n1, self.n2, separable=self.separable)
        LinearOperator.__init__(self, shape, shape)
    
    def __call__(self, x):
        return convolvefft(x, self.ker)
    
    def adjoint(self, x):
        return convolvefft(x, np.conjugate(self.ker))
    
    def gram(self, x):
        return convolvefft(x, np.conjugate(self.ker)*self.ker)
    
    def gram_resolvent(self, x, tau):
        return cg(lambda z: z + tau * self.gram(z), x)

class RandomMasking(LinearOperator):
    def __init__(self, shape, p):
        LinearOperator.__init__(self, shape, shape)
        self.total= np.prod(np.array(shape))
        self.no_zeros= p*self.total
        if len(shape)==2:
            self.shape=(shape[0], shape[1], 1)
        else:
            self.shape=shape
        self.H=np.ones(self.shape)
        counter=0
        while counter <= self.no_zeros:
            (i,j,k)=(np.random.randint(1,self.shape[0]+1),np.random.randint(1,self.shape[1]+1),np.random.randint(1,self.shape[2]+1))
            if self.H[i-1,j-1,k-1]==1:
                counter+=1
                self.H[i-1,j-1,k-1]=0
                
    def __call__(self, x):
        return self.H*x
        
    def adjoint(self, x):
        return self.H*x
    
    def gram(self, x):
        return self.H*self.H*x
    
    def gram_resolvent(self, x, tau):
        return cg(lambda z: z + tau * self.gram(z), x)
        
def heat_diffusion(y, m, gamma, scheme='continuous'):
    n1,n2=y.shape[0], y.shape[1]
    nu = (kernel('laplacian1_'),kernel('laplacian2_'))
    L = kernel2fft(nu, n1, n2, separable='sum')
    if scheme=='explicit':
        K_ee = (1 + gamma * L)**m
        x = convolvefft(y, K_ee)
        
    if scheme=='implicit':
        K_ie = 1 / (1 - gamma * L)**m
        x = convolvefft(y, K_ie)

    if scheme=='continuous':
        u, v = fftgrid(n1, n2)
        K_cs = np.exp(-(u**2 + v**2) / (4*gamma*m)) / (4*np.pi*gamma*m)
        K_cs = np.fft.fft2(K_cs, axes=(0, 1))
        x = convolvefft(y, K_cs)
    return x
        
def norm2(v, keepdims=True):
    if(len(v.shape)==3):
        v=np.reshape(v, (v.shape[0],v.shape[1],1, v.shape[2]))
    elif (len(v.shape)==1):
        v=np.reshape(v, (v.shape[0],v.shape[1],1))
    norm= lambda v1, v2: v1**2 + v2**2
    if(len(v.shape)==4):
        a= norm(v[:,:, 0, :], v[:,:,1, :])
    else:
        a= norm(v[:,:, 0], v[:,:,0])/2
    if len(a.shape)==3 and a.shape[2]==3:
        a=np.sum(a,axis=-1)
        if keepdims==True:
            a=np.reshape(a, (a.shape[0],a.shape[1],1,1 ))
    else:
        if keepdims==True:
            a=np.reshape(a, (a.shape[0],a.shape[1],1 ))
    return a  

def anisotropic_step(x, z, gamma, g, nusig, return_conductivity=False): 
    x_conv = convolve(x, nusig)
    alpha = g(norm2(grad(x_conv)))
    x = z + gamma * div(alpha * grad(z))
    if return_conductivity:
        return x, alpha
    else:
        return x
    
def anisotropic_diffusion(y, m, gamma, g=None, return_conductivity=False, scheme='explicit'):
    x = y
    if (len(x.shape)==3):
        C=3
    else:
        C=1
    if g==None:
        g= lambda u: 10 / (10 + 255*255*u/(np.sqrt(C)))
    nusig = kernel('gaussian', tau=0.2, s1=1, s2=1)
    for k in range(m):
        if scheme=='explicit':
            x, alpha= anisotropic_step(x, x, gamma, g, nusig, return_conductivity=return_conductivity)
        elif scheme=='implicit':
            x, alpha= cg(lambda z: rad_step(x, z, -gamma, g, nusig, return_conductivity=return_conductivity), x)
    if return_conductivity:
        return x, alpha
    else:
        return x