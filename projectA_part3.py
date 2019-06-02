""" Project A

COMPLETE THIS FILE

Your names here:

"""
import numpy as np
from .assignment6 import *

def cross_correlation(x, nu, boundary='periodical'):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv+=shift(x, k, l, boundary=boundary)*nu[k+s1,l+s2]
    return xconv

class Identity(LinearOperator):
    def __init__(self, ishape):
        self.shape=ishape
        oshape=ishape
        LinearOperator.__init__(self, ishape, oshape)
                
    def __call__(self, x):
        self.H= np.ones(self.ishape, dtype=int)
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
            self.shape=np.expand_dims(shape, axis=2)
        else:
            self.shape=shape
        self.H=np.ones(shape)
        counter=0
        while counter <= self.no_zeros:
            (i,j,k)=(np.random.randint(1,shape[0]+1),np.random.randint(1,shape[1]+1),np.random.randint(1,shape[2]+1))
            if self.H[i-1,j-1,k-1]==1:
                counter+=1
                self.H[i-1,j-1,k-1]=0
                
    def __call__(self, x):
        return self.H*x
        
    def adjoint(self, x):
        fft_H= np.fft.fft2(self.H, axes=(0,1))
        return np.real(np.fft.ifft2(cross_correlation(np.fft.fft2(x, axes=(0,1)), np.conjugate(fft_H)),axes=(0,1)))
    
    def gram(self, x):
        fft_H= np.fft.fft2(self.H, axes=(0,1))
        return np.real(np.fft.ifft2(cross_correlation(np.fft.fft2(x, axes=(0,1)), np.conjugate(fft_H)*fft_H),axes=(0,1)))
    
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
    
def anisotropic_diffusion(y, m, gamma, g=None, return_conductivity=False, scheme='explicit',variant =None):
    if variant is 'truly':
            x = y
            if len(y.shape)==3:
                C=3
            else:
                C=1
            if g==None:
                g= lambda u: 10 / (10 + 255*255*u/(np.sqrt(C)))
            nusig = im.kernel('gaussian', tau=0.2, s1=1, s2=1)
            nurho = (im.kernel(name ='gaussian1',tau = 0.5),im.kernel(name ='gaussian2',tau = 0.5))
            
            for k in range(m):
                if scheme=='explicit':
                    
                    x, alpha= truly_anisotropic_step(x, x, gamma, g, nusig,nurho, return_conductivity=return_conductivity)
                elif scheme=='implicit':
                    x, alpha= im.cg(lambda z: im1.rad_step(x, z, -gamma, g, nusig, return_conductivity=return_conductivity), x)
            if return_conductivity:
                return x, alpha
            else:
                return x
    if variant is None:
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
        
    
def tensorize(v, nurho):
    n1, n2, p = v.shape[:3]
    M = np.zeros((n1, n2, p, p))
    v_pq = np.zeros((n1,n2))
    for k in range(p):
        for l in range(p):
            if np.ndim(v) == 3:
                vv_pq = np.multiply(v[:,:,k],v[:,:,l])
            else:
                import pdb
#                 pdb.set_trace()
                vv_pq = np.sum(np.multiply(v[:,:,k,:],v[:,:,l,:]),axis=-1)
            #remember to sum across the three channels for the colour images
            M[:,:,k,l] = convolve(x = vv_pq,nu = nurho,boundary = 'periodical',separable= 'sum') 
    return M

def matrix_spectral_func(M, g):
    a = np.zeros_like(M)
    #find the eigenvalues of M along the last two dimension
    u,s,vh = np.linalg.svd(M, compute_uv=True)
    
    
    A = g(s)
    
    T = u @ (A[..., None] * vh)
    
    return T

def truly_anisotropic_step(x, z, gamma, g, nusig, nurho,
return_conductivity=False):
    
    
    #step 1
    v0 = im.grad(im.convolve(x = x,nu = nusig,boundary = 'periodical'))
    #step 2
    M = tensorize(v0,nurho)
#     import pdb
#     pdb.set_trace()
    n1,n2, p = M.shape[:3]
    #step 3
    T = matrix_spectral_func(M,g)
    #step 4

    v = np.zeros((n1,n2,p))
    if np.ndim(x) == 3:
        x0 = im.grad(x)
        v = (np.matmul(T,x0))
    else:
        x0 = im.grad(x).reshape(n1,n2,p,1)
        v = (np.matmul(T,x0)).reshape(n1,n2,p)

#     pdb.set_trace()
    #finallly
    x =z + gamma*im.div(v)

    if return_conductivity:
        return x, T
    else:
        return x
