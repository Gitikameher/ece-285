""" Provided code for assignments and projects of UCSD ECE285-IVR.

DO NOT MODIFY THIS FILE

Copyright Charles Deledalle, 2019
"""

import numpy as np
import numpy.fft as nf
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from .assignment5 import *


#############
## Display ##
#############


# Set matplotlib parameters for figure display
params = {'font.family': ['serif'],
          'font.size': 7,
          'legend.fontsize': 7,
          'axes.labelsize': 6,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'figure.figsize': (5, 3),
          'figure.autolayout': True}
matplotlib.rcParams.update(params)
del params


def show(x, ax=None, vmin=0, vmax=1, *args, **kargs):
    """ Display an image

    Like `~matplotlib.pyplot.imshow` but without showing axes, and
    the range [vmin, vmax] is also effective for RGB images.
    Use grayscale colormap for scalar images.

    Parameters
    ----------
    x : array-like
        An image, float, of shapes (M, N), (M, N, 3) or (M, N, 4)
    ax : a `~matplotlib.axes.Axes` object, optional
        Axes on which to display the image. If not given, current instance.
    vmin, vmax: scalars, optional
        Define the data range that the colormap covers.
        For scalar images, black is vmin and white is vmax.
        For RGB images, black is [vmin, vmin, vmin] and red is [vmax, vmin, vmin].
        By default the range is [0, 1].

    Returns
    -------
    image : `~matplotlib.image.AxesImage`

    Notes
    -----
    All other optional arguments are transfered to matplotlib.pyplot.imshow.

    Copyright Charles Deledalle, 2019
    """

    ax = plt.gca() if ax is None else ax
    x = x.copy().squeeze()
    if vmin is not None:
        x[x < vmin] = vmin
    if vmax is not None:
        x[x > vmax] = vmax
    if x.ndim == 2:
        h = ax.imshow(x, cmap=plt.gray(), vmin=vmin, vmax=vmax, *args, **kargs)
    else:
        vmin = x.min() if vmin is None else vmin
        vmax = x.max() if vmax is None else vmax
        x = (x - vmin) / (vmax - vmin)
        h = ax.imshow(x, vmin=0, vmax=1, *args, **kargs)
    ax.axis('off')
    return h


######################
## Fourier analysis ##
######################


def showfft(x, ax=None, vmin=None, vmax=None, apply_fft=False, apply_log=False, *args, **kargs):
    """ Display the amplitude of an image spectrum (Fourier transform).

    The zero-frequency is centered. Spectrum is normalized by the image size.
    Both axes are numbered by their corresponding frequencies. Grid is displayed.
    By default, the color map range is optimized for visualization.

    Parameters
    ----------
    x : array-like
        The 2d spectrum of an image, complex, of shapes (M, N), (M, N, 3) or (M, N, 4)
    ax : a `~matplotlib.axes.Axes` object, optional
        Axes on which to display the image spectrum. If not given, current instance.
    apply_fft: boolean, optional
        If True, input x is replaced by `~numpy.fft.fft2(x)`. Default, False.
    apply_log: boolean, optional
        If True, the log of the amplitude is displayed instead.
    vmin, vmax: scalars, optional
        Define the data range that the colormap covers.
        If apply_log=False, by default the range is [0, MAX] where MAX is the maximum
        value of the amplitude of the spectrum. If apply_log=True, by default the
        range is [LMAX-16, LMAX] where LMAX is the maximum value of the log amplitude
        of the spectrum.

    Returns
    -------
    image : `~matplotlib.image.AxesImage`

    Notes
    -----
    All other optional arguments are transfered to matplotlib.pyplot.imshow.

    Copyright Charles Deledalle, 2019
    """

    ax = plt.gca() if ax is None else ax
    n1, n2 = x.shape[:2]
    xpos = np.linspace(0, n2-1, n2)
    xfreq = nf.fftshift(nf.fftfreq(n2, d=1./n2))
    ypos = np.linspace(0, n1-1, n1)
    yfreq = nf.fftshift(nf.fftfreq(n1, d=1./n1))
    x[np.isinf(x)] = x[np.logical_not(np.isinf(x))].max()

    def on_lims_change(axes):
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        xsubidx = np.linspace(xlim[0]+1.5, xlim[1]-.5, 9).astype(np.int)
        ysubidx = np.linspace(ylim[0]-.5, ylim[1]+1.5, 9).astype(np.int)
        axes.set_xticks([xpos[i] for i in xsubidx])
        axes.set_yticks([ypos[i] for i in ysubidx])
        axes.set_xticklabels(['%d' % xfreq[i] for i in xsubidx])
        axes.set_yticklabels(['%d' % yfreq[i] for i in ysubidx])

    data = np.abs(nf.fft2(x, axes=(0, 1)) if apply_fft else x) / (n1 * n2)
    data = np.log(data) if apply_log else data
    h = show(nf.fftshift(data),
             vmin=(data.max() - 16 if apply_log else 0) if vmin is None else vmin,
             vmax=data.max() if vmax is None else vmax, ax=ax,
             * args, **kargs)
    ax.axis('on')
    ax.callbacks.connect('xlim_changed', on_lims_change)
    ax.callbacks.connect('ylim_changed', on_lims_change)
    on_lims_change(ax)
    ax.grid(color='r', alpha=.4, linestyle='-', linewidth=.5)
    return h


def fftgrid(n1, n2):
    """ Create a 2d grid of spectral frequencies.

    Parameters
    ----------
    n1, n2 : integers
        The size of the grid.

    Returns
    -------
    u, v : arrays of shape (n1, n2)
        Elements u[i, j] and v[i, j] indicates the corresponding 2d frequency of
        the index [i, j] in an array returned by `~numpy.fft.fft2`.

    Copyright Charles Deledalle, 2019
    """

    f1 = nf.fftfreq(n1, d=1./n1)
    f2 = nf.fftfreq(n2, d=1./n2)
    u, v = np.meshgrid(f1, f2, indexing='ij')
    return u, v


####################
## Linear algebra ##
####################


def cg(A, b, x0=None, eps=1e-3, maxit=1000):
    """ Solve A(x) = b by conjugate gradient

    Parameters
    ----------
    A : function
        Function in the left hand side of A(x) = b.
        Input x, output A(x) and b should be numpy arrays, float, with the same
        shape. Function A must satisfy the two properties:

        - symmetric definite non-negative function:
            - <x, A(y)> = <y, A(x)>      for all x, y
            - <x, A(x)> >= 0             for all x
        - invertible for b (a solution must exists).

    b : array_like
        Right hand side of A(x) = b.
        b must have the same shape as the input and ouput of A.
    x0 : array_like, optional
        Initialization. If None, set to zero. Default None
    eps : scalar, optional
        Precision at wich to stop. Default 1e-3
    maxit: integer, otpional
        Maximum number of iterations. Default 1000.

    Note
    ----
    If A is also symmetric definite positive, <x, A(x)> > 0, then it is
    necessarily invertible (a solution exists and will be found).

    If A is non-invertible and x0=None, the Moore-Penrose pseudo inverse is
    returned. But for arbitrary x0, CG might not converge.

    If A is not symmetric, you can solve instead AT(A(x)) = AT(b) with:
    cg(lambda x: AT(A(x)), AT(b)) where AT is the adjoint of A satisfying:
    <y, A(x)> = <AT(y), x> for all x and y of suitable shapes. Solutions will
    be minimizers of ||y - A(x)||**2.

    Returns
    -------
    x : array_like
        The solution of A(x) = b.

    Copyright Charles Deledalle, 2019
    """

    x = np.zeros(b.shape) if x0 is None else x0
    i = 0
    r = b - A(x)
    d = r
    dn = np.sum(r ** 2)
    d0 = dn
    while i < maxit and dn > eps ** 2 * d0:
        q = A(d)
        dq = np.sum(d * q)
        if dq == 0:
            warnings.warn('Is the operator invertible?', RuntimeWarning)
            break
        a = dn / dq
        x = x + a * d
        r = r - a * q
        do = dn
        dn = np.sum(r ** 2)
        if do == 0:
            warnings.warn('Is the operator invertible?', RuntimeWarning)
            break
        b = dn / do
        d = r + b * d
        i = i + 1
    return x


###############
## Operators ##
###############


class LinearOperator(ABC):
    """ Abstract class for Linear Operators

    Methods
    -------
    LinearOperator(ishape, oshape=None) : constructor
        Create an instance of a linear operator A taking inputs of shape ishape
        and producing output of shape oshape. If oshape=None, then
        oshape=ishape.
    *(x) : abstract method
        Apply the linear operator A on object x (x must have shape ishape).
    *.adjoint(z) : abstract method
        Apply the adjoint A* on object z (z must have shape oshape).
    *.gram(x) : abstract method
        Apply the gram operator A*A on object x (x must have shape ishape).
    *.gram_resolvant(x, tau) : abstract method
        Apply the resolvent of the gram operator (Id + tau A*A)^-1 on object x
        (x must have shape ishape).
    *.norm2() : method
        Return an approximation of the spectral norm of A: ||A||_2
    *.normfro() : method
        Return an approximation of the Frobenius norm of A: ||A||_F

    Properties:
    -----------
    *.ishape : property
        Return the shape of the input of the operator A
    *.oshape : property
        Return the shape of the output of the operator A

    Copyright Charles Deledalle, 2019
    """

    def __init__(self, ishape, oshape=None):
        if oshape is None:
            oshape = ishape
        self.__ishape = ishape
        self.__oshape = oshape
        self._norm2 = None
        self._normfro = None

    @property
    def ishape(self):
        return self.__ishape

    @property
    def oshape(self):
        return self.__oshape

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def adjoint(self, x):
        pass

    @abstractmethod
    def gram(self, x):
        pass

    @abstractmethod
    def gram_resolvent(self, x, tau):
        pass

    def norm2(self):
        if self._norm2 is None:
            K = 100
            x = np.random.randn(*self.ishape)
            for k in range(K):
                y = self.gram(x)
                x = y / np.sqrt((y**2).sum())
            self._norm2 = np.sqrt(np.sqrt((y**2).sum()))
        return self._norm2

    def normfro(self, normalize=False):
        if self._normfro is None:
            K = 100
            n = 0
            for k in range(K):
                x = 2 * np.random.binomial(1, .5, size=self.ishape) - 1
                n += (x * self.gram(x)).sum()
            n /= K
            self._normfro = np.sqrt(n)
        if normalize:
            return self._normfro / np.sqrt(np.array(self.ishape).prod())
        else:
            return self._normfro


class Grad(LinearOperator):
    """ Gradient Operator (inherit from LinearOperator)

    Implement/overload the following methods
    ----------------------------------------
    Grad(ishape, boundary='periodical') : constructor
        Create an instance of the 2d discrete gradient operator A=nabla for an
        input image of shape ishape=(n1, n2, ...) producing an output vector
        field of shape oshape=(n1, n2, 2, ...). The leading dimensions can be
        anything: empty, 3 for RGB, etc. Boundary conditions must be specified
        by the optional argument boundary that can take the values 'periodical',
        'mirror', 'extension' or 'zeropadding'.
    *(x) : method
        Apply the gradient on image x (x must have shape ishape). It returns a
        vector field of shape oshape=(n1, n2, 2, ...)
    *.adjoint(z) : method
        Apply the adjoint -div on object z (z must have shape oshape).
    *.gram(x) : method
        Apply the gram operator -laplacian on object x (x must have shape
        ishape).
    *.gram_resolvant(x, tau) : method
        Apply the resolvent of the gram operator (Id - tau laplacian)^-1 on
        object x (x must have shape ishape).

    Note
    ----
    This class won't work before Assignments 2 to 5 will be completed.

    Copyright Charles Deledalle, 2019
    """

    def __init__(self, ishape, boundary='periodical'):
        n1, n2 = ishape[:2]
        oshape = list(ishape)
        oshape.insert(2, 2)
        oshape = tuple(oshape)
        LinearOperator.__init__(self, ishape, oshape)
        self._boundary = boundary
        if boundary is 'periodical':
            lplcn_nu = (kernel('laplacian1'), kernel('laplacian2'))
            self._lplcn_lbd = kernel2fft(lplcn_nu, n1, n2, separable='sum')

    def __call__(self, x):
        return grad(x, boundary=self._boundary)

    def adjoint(self, x):
        return -div(x, boundary=self._boundary)

    def gram(self, x):
        return -laplacian(x, boundary=self._boundary)

    def gram_resolvent(self, x, tau):
        if self._boundary is 'periodical':
            # We can do the inversion in the Fourier domain
            res_lbd = 1 / (1 - tau * self._lplcn_lbd)
            return convolvefft(x, res_lbd)
        else:
            # We can do the inversion by Conjugate Gradient
            return cg(lambda z: z + tau * self.gram(z), x)


######################
## Wavelet analysis ##
######################


def dtw_crop(x, J):
    """ Crop an image to the largest dimension compatible with the 2d discrete
    wavelet transform (DWT) with J scales.

    Parameters
    ----------
    x : array-like
        An image, float, of shapes (M, N), (M, N, 3) or (M, N, 4)
    J : integer
        The number of scales in the DWT.

    Returns
    -------
    y : array-like
        The cropped image, float, of shapes (M, N), (M, N, 3) or (M, N, 4)

    Copyright Charles Deledalle, 2019
    """

    n1, n2 = x.shape[:2]
    r1 = np.mod(n1, 2**J)
    r2 = np.mod(n2, 2**J)
    if r1 > 0:
        x = x[int(r1/2):-(r1-int(r1/2)), :]
    if r2 > 0:
        x = x[:, int(r2/2):-(r2-int(r2/2))]
    return x


def dwt_power(n1, n2, J, ndim=3):
    """ Create the power map for the 2d discrete wavelet transform (DWT) with J
    scales.

    Parameters
    ----------
    n1, n2 : integers
        The size of the DWT.
    J : integer
        The number of scales in the DWT.
    ndim : integer, optional
        Indicate the number of dimensions (including the two spatial dimension).
        For grayscale image use `ndim=2`, for color use `ndim=3`, for tensors
        `ndim=4` and so on.

    Returns
    -------
    p : array-like of shape (n1, n2) or (n1, n2, 1, 1, ...) depending of ndim.
        Entries p[x, y] = 2**(j-1) if x, y is in a detailed subband of scale j
        Entries p[x, y] = 2**J if x, y is in the coarse subband

    Copyright Charles Deledalle, 2019
    """

    if J == 0:
        return np.ones((n1, n2, *[1] * (ndim - 2)))
    m1, m2 = int(n1/2), int(n2/2)
    c = 2 * dwt_power(m1, m2, J - 1, ndim=ndim)
    de = np.ones((m1, m2, *[1] * (ndim - 2)))
    p = np.concatenate((np.concatenate((c, de), axis=0),
                        np.concatenate((de, de), axis=0)), axis=1)
    return p


def udwt_power(J, ndim=3):
    """ Create the power coefficients for the 2d undecimated discrete wavelet
    transform (UDWT) with J scales.

    Parameters
    ----------
    J : integer
        The number of scales in the DWT.
    ndim : integer, optional
        Indicate the number of dimensions (including the two spatial dimension).
        For grayscale image use `ndim=2`, for color use `ndim=3`, for tensors
        `ndim=4` and so on.

    Returns
    -------
    p : array-like of shape (1, 1, 3 * J + 1) or (1, 1, 3 * J + 1, 1, 1, ...)
        depending of ndim.
        Entries p[:, :, k] = 2 * 4**(j-1) if channel k is a subband of scale j
        Entries p[:, :, k] = 4**j if channel k is the coarse subband

    Copyright Charles Deledalle, 2019
    """

    p = ((4)**J, )
    for j in range(1, J+1):
        p += ((4)**(J - j), ) * 3
    p = np.array(p, dtype=np.float32).reshape(1, 1, -1, *[1] * (ndim - 2))
    p[:, :, 1:] *= 2
    return p


def showdwt(z, J, ax=None, vmin=None, vmax=None, apply_norm=True, *args, **kargs):
    """ Display the 2D DWT of an image.

    By default, coefficients are normalized for best display.
    Subbands are indicated by a diadic grid.

    Parameters
    ----------
    z : array-like
        The 2d DWT coefficients of an image, float, of shapes (M, N), (M, N, 3) or (M, N, 4)
    J : integer
        The number of scales in the DWT.
    ax : a `~matplotlib.axes.Axes` object, optional
        Axes on which to display the image spectrum. If not given, current instance.
    apply_norm: boolean, optional
        If True, normalize z by the DWT power and center the coarse scale to 0.
        Default, True.
    vmin, vmax: scalars, optional
        Define the data range that the colormap covers, basically the one of the
        corresponding image.

    Returns
    -------
    image : `~matplotlib.image.AxesImage`

    Notes
    -----
    All other optional arguments are transfered to matplotlib.pyplot.imshow.

    Copyright Charles Deledalle, 2019
    """

    ax = plt.gca() if ax is None else ax
    n1, n2 = z.shape[:2]
    z = z.copy()
    if apply_norm:
        z = z / dwt_power(*z.shape[:2], J).reshape(n1, n2, *[1] * (z.ndim - 2))
        if vmin is None:
            vmin = z[:int(n1/2**J), :int(n2/2**J)].min()
        if vmax is None:
            vmax = z[:int(n1/2**J), :int(n2/2**J)].max()
        z[:int(n1/2**J), :int(n2/2**J)] -= vmin
        z *= 2 / (vmax - vmin)
        z[:int(n1/2**J), :int(n2/2**J)] -= 1
        h = show(z, ax=ax, vmin=-1, vmax=1, *args, **kargs)
    else:
        h = show(z, ax=ax, vmin=vmin, vmax=vmax, *args, **kargs)
    for j in range(1, J+1):
        ax.plot([-.5, n2 / 2**(j - 1) - .5], [n1 / 2**j - .5, n1 / 2**j - .5], 'r',
                alpha=.4, linestyle='-', linewidth=.5)
    for j in range(1, J+1):
        ax.plot([n2 / 2**j - .5, n2 / 2**j - .5], [-.5, n1 / 2**(j - 1) - .5], 'r',
                alpha=.4, linestyle='-', linewidth=.5)
    return h


def wavelet(name, d=2):
    """ Create low- and high-pass wavelet convolution filters.

    Parameters
    ----------
    name : string
        Name of the wavelet. Choices are 'haar', 'db1', 'db2', 'db4', 'db8',
        'sym4', 'coif4'.
    d : integer, optional
        The dimension of the signal on which this wavelet will be applied: d=1
        for 1d signals, d=2 for images (default) and so on.

    Returns
    -------
    h, g : array-like
        Two one-dimensional arrays with shape (n,), (n, 1), (n, 1, 1), etc,
        depending on d. The length n depends on the wavelet but is necessarily
        odd. The two arrays define two periodical convolution kernels
        (compatible with `~imagetools.convolve`) corresponding to the high- and
        low-pass wavelet filters, respectively.

    Copyright Charles Deledalle, 2019
    """

    if name in ('haar', 'db1'):
        h = np.array([-1, 1])
    if name is 'db2':
        h = np.array([1, np.sqrt(3), -(3 + 2 * np.sqrt(3)), 2 + np.sqrt(3)])
    if name is 'db4':
        h = np.array(
            [-0.230377813308855230, +0.714846570552541500, -0.630880767929590400,
             -0.027983769416983850, +0.187034811718881140, +0.030841381835986965,
             -0.032883011666982945, -0.010597401784997278])
    if name is 'db8':
        h = np.array(
            [-0.0544158422, +0.3128715909, -0.6756307363, +0.5853546837,
             +0.0158291053, -0.2840155430, -0.0004724846, +0.1287474266,
             +0.0173693010, -0.0440882539, -0.0139810279, +0.0087460940,
             +0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])
    if name is 'sym4':
        h = np.array(
            [-0.03222310060404270, -0.012603967262037833, +0.09921954357684722,
             +0.29785779560527736, -0.803738751805916100, +0.49761866763201545,
             +0.02963552764599851, -0.075765714789273330])
    if name is 'coif4':
        h = np.array(
            [-0.00089231366858231460, -0.00162949201260173260, +0.00734616632764209350,
             +0.01606894396477634800, -0.02668230015605307200, -0.08126669968087875000,
             +0.05607731331675481000, +0.41530840703043026000, -0.78223893092049900000,
             +0.43438605649146850000, +0.06662747426342504000, -0.09622044203398798000,
             -0.03933442712333749000, +0.02508226184486409700, +0.01521173152794625900,
             -0.00565828668661072000, -0.00375143615727845700, +0.00126656192929894450,
             +0.00058902075624433830, -0.00025997455248771324, -6.2339034461007130e-05,
             +3.1229875865345646e-05, +3.2596802368833675e-06, -1.7849850030882614e-06])
    h = h / np.sqrt(np.sum(h**2))
    g = (-1)**(1 + np.arange(h.shape[0])) * h[::-1]
    h = np.concatenate((h, np.array([0.])))
    g = np.concatenate((g, np.array([0.])))
    h = h.reshape(-1, *[1] * (d - 1))
    g = g.reshape(-1, *[1] * (d - 1))
    return h, g
