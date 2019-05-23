""" Project A

COMPLETE THIS FILE

Your names here:

"""
import numpy as np
from .assignment6 import *

class Identity(LinearOperator):
    def __init__(self,shape):
        self.shape=shape
    
    def __call__(self, x):
        self.H= np.ones(self.shape, dtype=int)
        return convolvefft(x, self.H)
    
    def adjoint(self, x):
        return convolvefft(x, np.conjugate(self.H))
    
    def gram(self, x):
        return convolvefft(x, np.conjugate(self.H)*self.H)
    
    def gram_resolvent(self, x, tau):
        return cg(lambda z: z + tau * self.gram(z), x)
        
class Convolution(LinearOperator):
    def __init__(self,shape, nu, separable=None):
        self.n1=shape[0];self.n2=shape[1]
        self.nu=nu
        self.separable=separable
        self.ker=kernel2fft(self.nu, self.n1, self.n2, separable=self.separable)
    
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
        self.total= np.prod(np.array(shape))
        self.no_zeros= p*total
        if len(shape)==2:
            self.shape=np.expand_dims(shape, axis=2)
        else:
            self.shape=shape
        self.H=np.ones(shape)
        counter=0
        while counter <= no_zeros:
            (i,j,k)=(np.random.randint(1,shape[0]+1),np.random.randint(1,shape[1]+1),np.random.randint(1,shape[2]+1))
            if self.H[i-1,j-1,k-1]==1:
                counter+=1
                self.H[i-1,j-1,k-1]=0
                
    def __call__(self, x):
        return self.H*x
        
        
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
    a= (np.absolute(v))**2
    if len(v.shape)==3 and v.shape[2]==3:
        a=np.sum(a,axis=-1)
        if keepdims==True:
            a=np.reshape(a, (a.shape[0],a.shape[1],1,1 ))
    else:
        if keepdims==True:
            a=np.reshape(a, (a.shape[0],a.shape[1],1 ))
    return a  

def anisotropic_step(x, z, gamma, g, nusig, return_conductivity=False): 
    x_conv = convolve(x, nusig)
    temp=grad(x_conv)
    alpha = g(np.stack((norm2(temp[:,:,0, :]), norm2(temp[:,:,1, :])), axis=2))
    x = z + gamma * div(alpha.reshape(alpha.shape[0],alpha.shape[1],alpha.shape[2],alpha.shape[3]) * grad(z))
    if return_conductivity:
        return x, alpha
    else:
        return x
    
def anisotropic_diffusion(y, m, gamma, g=None, return_conductivity=False, scheme='explicit'):
    x = y
    if len(y.shape)==3:
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