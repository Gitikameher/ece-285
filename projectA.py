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
            M[:,:,k,l] = convolve(x = vv_pq,nu = nurho,boundary = 'periodical') 
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
    v0 = grad(convolve(x,nu = nusig,boundary = 'periodical'))
    #step 2
    M = tensorize(v0,nurho)

    n1,n2, p = M.shape[:3]
    #step 3
    T = matrix_spectral_func(M,g)
    #step 4

    v = np.zeros((n1,n2,p))
    if np.ndim(x) == 3:
        x0 = grad(z)
        v = (np.matmul(T,x0))
    else:
        x0 = grad(z).reshape(n1,n2,p,1)
        v = (np.matmul(T,x0)).reshape(n1,n2,p)
    #finallly
    x =z + gamma*div(v)

    if return_conductivity:
        return x, T
    else:
        return x    
    
def anisotropic_diffusion(y, m, gamma, g=None, return_conductivity=False, scheme='explicit',variant =None):
    if variant is 'truly':
        if scheme=='implicit':
            return_conductivity=False
        x = y
        z=y
        if len(y.shape)==3:
            C=3
        else:
            C=1
        if g==None:
            g= lambda u: 10 / (10 + 255*255*u/(np.sqrt(C)))
        nusig = kernel('gaussian', tau=0.2, s1=1, s2=1)
        nurho = kernel(name ='gaussian',tau = 0.5, s1=1,s2=1)

        for k in range(m):
            if scheme=='explicit':

                x, alpha= truly_anisotropic_step(x, x, gamma, g, nusig,nurho, return_conductivity=return_conductivity)
            elif scheme=='implicit':
                
                x = cg(lambda z: truly_anisotropic_step(x, z, -gamma, g, nusig,nurho, return_conductivity=return_conductivity), x)
        if return_conductivity:
            return x, alpha
        else:
            return x
    if variant is None:
               
        if scheme=='implicit':
            return_conductivity=False
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
                if return_conductivity:
                    x, alpha= anisotropic_step(x, x, gamma, g, nusig, return_conductivity=return_conductivity)
                else:
                    x = anisotropic_step(x, x, gamma, g, nusig, return_conductivity=return_conductivity)
            elif scheme=='implicit':

                x= cg(lambda z: anisotropic_step(x, z, -gamma, g, nusig, return_conductivity=return_conductivity), x)
        if return_conductivity:
            return x, alpha
        else:
            return x
 