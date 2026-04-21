#### Code adapted from Lionel Moisan 
#### https://helios2.mi.parisdescartes.fr/~moisan/sharpness/

import numpy as np 
import math 
from scipy.special import erfc
from copy import deepcopy 
from skimage.transform import resize
import scipy

def perdecomp(U):
	#code taken from the MVA course by L Moisan
	u=deepcopy(U)
	ny, nx = u.shape
	X = np.arange(nx)
	Y = np.arange(ny)
	v = np.zeros((ny, nx))
	v[ 0,X] = u[0,X] - u[-1,X] 
	v[-1,X] = -v[0,X]
	v[Y, 0] = v[Y, 0] + u[Y,0] - u[Y,-1]
	v[Y,-1] = v[Y,-1] - u[Y,0] + u[Y,-1]
	fx = np.tile(np.cos(2.*math.pi*X/nx), (ny,1))
	fy = np.tile(np.cos(2.*math.pi*Y/ny), (nx,1)).T
	fx[0,0] = 0 # avoid division by 0 in the line below
	s = np.real(np.fft.ifft2(np.fft.fft2(v)*0.5/(2.-fx-fy)))
	p = u-s
	#where p is called the periodic component of u and s is the smooth component of u.
	return p, s

def dequant(u):
	ny,nx = u.shape
	mx,my=math.floor(nx/2),math.floor(ny/2)
	Tx=np.exp(-1j*math.pi/nx*((np.arange(mx,mx+nx-1))%nx-mx))
	Ty=np.exp(-1j*math.pi/ny*((np.arange(my,my+ny-1))%ny-my))
	v=np.real(np.fft.ifft2(np.fft.fft2(u)*(Ty@Tx.T)))
	return v


def logerfc(x):
    T = (x>20)
    if T.sum()>0:
        y=deepcopy(x)
        z=x**(-2)
        s=1
        for k in np.arange(8,1,-1):
            s = 1-(k-0.5)*z*s
        y = -0.5*np.log(math.pi)-x**2+np.log(s/x)
    else:
        y=np.log(erfc(x))
    return y 

def s_index(U):
	u=deepcopy(U)
	u,_=perdecomp(u)
	u=dequant(u)
	u=u.astype('double')
	ny,nx = u.shape
	ind_x,ind_y=np.zeros(nx,dtype=int),np.zeros(ny,dtype=int)
	ind_x[:-1]=np.arange(1,nx)
	ind_x[-1]=0
	ind_y[:-1]=np.arange(1,ny)
	ind_y[-1]=0
	gx = u[:,ind_x]-u
	gy = u[ind_y,:]-u
	tv = ((np.abs(gx)+np.abs(gy))).sum()
	fu=np.fft.fft2(u)
	p=np.arange(0,nx)
	q=np.reshape(np.arange(0,ny),(1,ny))
    #repmat(a, m, n) is tile(a, (m, n)). 
	P=np.tile(p,(ny,1))*2*math.pi/nx
	Q=np.tile(np.conjugate(q.T),(1,nx))*2*math.pi/ny
	fgx2 = fu*np.sin(P/2)
	fgx2 = np.real(4*fgx2*np.conjugate(fgx2))
	fgy2 = fu*np.sin(Q/2)
	fgy2 = np.real(4*fgy2*np.conjugate(fgy2))
	fgxx2=(fgx2**2).sum()
	fgyy2=(fgy2**2).sum()
	fgxy2=(fgx2*fgy2).sum()
	vara = 0
	axx=(gx**2).sum()
	if axx>0:
		vara=vara+fgxx2/axx
	ayy=(gy**2).sum()
	if ayy>0:
		vara = vara+fgyy2/ayy
	axy=np.sqrt(axx*ayy)
	if axy>0:
		vara = vara+2*fgxy2/axy
	vara = vara/(math.pi*nx*ny)
	if vara>0: #t = ( E(TV)-tv )/sqrt(vara)
		t = ((np.sqrt(axx)+np.sqrt(ayy))*np.sqrt(2*nx*ny/math.pi) - tv )/np.sqrt(vara)
		s = -logerfc(t/np.sqrt(2))/np.log(10)+np.log10(2)
	else:
		s=0
	return s
def mu_lower_bound(Yh,Ym):
    def crude_solution(Yh,Ym):
        L=Yh.shape[0]
        M,N=Ym.shape[1],Ym.shape[2]
        crude=np.zeros((L,M,N))
        for chan in range(L):
            crude[chan,:,:]=resize(Yh[chan,:,:],(M,N), anti_aliasing=True)
        return crude
    Zhat=crude_solution(Yh,Ym)
    def calc_sigma(Y):
        lowpass = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            lowpass[i,:,:] = scipy.ndimage.gaussian_filter(Y[i,:,:], sigma = 5)
        gauss_highpass = Y - lowpass
        return 1.4826 * np.mean(scipy.stats.median_abs_deviation(gauss_highpass, axis=(1,2)))
    sigma_m=calc_sigma(Ym)
    alpha=np.linalg.norm(Zhat)
    return sigma_m**2/alpha**2
