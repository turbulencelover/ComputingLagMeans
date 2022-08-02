import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft2, ifft2
import scipy.io as spio
import time
from scipy.interpolate import RectBivariateSpline

# %% ============= About this code ==========================================
# % This code solves Shallow-Water (SW) using a pseudo-spectral method 
# % The equations are solved for variables:
# % (u,v)=(vel. components) and h = displacement of free surface
# % The SW equations are solved in the dimensionless form
# % In tandem with the SW equations, the Lagrangian means are calculated
# % using the strategy 2 of Kafiabad & Vanneste (2022)
# % in the current version the Lag mean of vorticity and height field are 
# % computed but the code can be easily used to compute any Lag means
# % Developed by Hossein Kafiabad 2022 (kafiabad@gmail.com) 
# % *************************************************************************
# % Main variables: 
# % uk, vk: velocities in Fourier space -- ur, vr: velocities in real space
# % hk: height in Fourier space -- hr: height in real space
# % zk: vorticity in Fourier space -- zr: vorticity in real space
# % *************************************************************************
# % Side notes:
# % 1) In this code we want to store the instantaneous fields at the mid of 
# % averaging interval. Hence, we compute the partial means in the first half
# % of interval, then store the instantaneous values (e.g. for vorticity),
# % and finally continue with the second half of the interval 
# % (this is more efficient than leave an if statement inside one time loop)
# % 2) For interpolation we padd the fields by adding the first row/column to
# % the matrices after the last row/column. This is necessary for
# % interpolation. For example, consider the periodic domain of [0, 10] which 
# % is discretised by 10 points. The function at x = 0 and x= 10 is the same. 
# % So in the pseudo-spectral code we use the disretised points 0,1,2 ..,9 as
# % the information at t=10 is redundant. However, if we need to interpolate
# % and the desired point is between 9 and 10 (like x=9.5), the interpolation
# % function cannot handle it as 9.5 does not exist between two grid points.
# % The padding throughout this script will take care of that.


## ========================== Parameters ==========================
Re = 3.84e3
Ro = 0.1
Fr = 0.5

Nx = 128
Ny = 128
dt = 2.5e-4
Tend = 2.2
Tave = Tend/2
Nt = int(Tend/dt)
Nt_hf = int(np.floor(Tave/dt))


## ========================== Spatial Variables + Wavenumbers ==========================
#  +++++++++++++++++++++++++++++++++++++++++++
#  1) in Real Space

# grid spacing in real space
dx = 2*np.pi/Nx
dy = 2*np.pi/Ny
# grid points in real space
x = (np.arange(0,Nx))*dx
y = (np.arange(0,Ny))*dy
# meshgrid of the points
xx, yy = np.meshgrid(x,y)
#  padding xx and yy only for interpolation (according to periodic BC)
#  See the side note 2) at the beginning of the script
x_padd = (np.arange(0,Nx+1))*dx
y_padd = (np.arange(0,Ny+1))*dy
xx_padd, yy_padd = np.meshgrid(x_padd,y_padd)

#  +++++++++++++++++++++++++++++++++++++++++++
#  1) in Fourier Space
Kx = np.zeros(Nx)
Kx[:int(Nx/2)] = np.arange(int(Nx/2))
Kx[int(Nx/2):] = np.arange(-int(Nx/2),0)
Ky = np.zeros(Ny)
Ky[:int(Ny/2)] = np.arange(int(Ny/2))
Ky[int(Ny/2):] = np.arange(-int(Ny/2),0)
Kxx, Kyy = np.meshgrid(Kx,Ky)
#  k2 is used for dissipation term i.e. -nu*k^2 
#  (we dont have issue with k2 =0)
#  k2poisson is for solving poisson equation
k2 = Kxx**2+Kyy**2
k2poisson = Kxx**2+Kyy**2
k2poisson[0,0] = 1

#  de-aliasing mask: forces the nonlinear terms for kx,ky>2/3 to be zero
#  depending on the problem and the type of dissipation can be relaxed ...
L = np.ones(np.shape(k2poisson))
for i in range(Nx):
    for j in range(Ny):
        if abs(Kxx[j, i]) > max(Kx)*2./3.:
            L[j, i] = 0
        elif abs(Kyy[j, i]) > max(Ky)*2./3.:
            L[j, i] = 0
            


## ========================== Initial Condition ==========================   
#  -----------------------------------------------------------------------
#  ***** 1) Geostrophic flow: *****
#  consist of a turbulent distribution of vortices that
#  is a solution of 2D Navier Stokes & stored in'uvk_2Dturbulence_256.mat'
ur = spio.loadmat('uvr_2Dturbulence_256.mat')['ur']
vr = spio.loadmat('uvr_2Dturbulence_256.mat')['vr']
ur = ur[0:-1:2,0:-1:2]
vr = vr[0:-1:2,0:-1:2]
uk = fft2(ur)
vk = fft2(vr)
zk = 1j*Kxx*vk-1j*Kyy*uk
#  Find the associated height field such that it would be in geostrophic
#  balance with the velocity fields
hk = -(Ro/Fr**2)*zk/k2poisson
hr = 1 + np.real(ifft2(hk))


#  ***** 2) Poincare wave: *****
Uw = 1.8
kw = 1
omeg = np.sqrt(Ro**-2+Fr**-2*kw**2)
ur_wave = -Uw*np.cos(kw*xx)
vr_wave = -Uw/omeg/Ro*np.sin(kw*xx)
hr_wave = -Uw/omeg*np.cos(kw*xx)

ur += ur_wave
vr += vr_wave
hr += hr_wave
uk = fft2(ur)
vk = fft2(vr)
hk = fft2(hr)

# zr = np.real(ifft2(zk))
# plt.pcolor(xx,yy,zr, cmap = 'RdBu', vmin=-0.83, vmax=0.83)
# plt.show()

## =================== initialising the Lag mean field quantities ===================
#  the initial value for LM quantities that are going to be updated in time
zk = 1j*Kxx*vk-1j*Kyy*uk
zr = np.real(ifft2(zk))
z_EMr = zr
h_EMr = hr
z_LMk = np.zeros(zk.shape)
h_LMk = np.zeros(hk.shape)
Xixr = np.zeros(xx.shape)
Xiyr = np.zeros(yy.shape)
Xixk = np.zeros(Kxx.shape)
Xiyk = np.zeros(Kyy.shape)
X_in = xx
Y_in = yy
uL_RHSk = uk;
vL_RHSk = vk;
zL_RHSk = zk;
hL_RHSk = hk;

#---------------------------------------------------------------
#  Calculate the viscosity coefficients out of the loop (for numerical efficiency)
Cc = np.exp(-dt/Re*k2)


#---------------- padding and interpolation functions ----------------#
def interp_fft(x_padd, y_padd, qr, X_in, Y_in): 
    Q_padd = np.zeros((qr.shape[0]+1,qr.shape[1]+1))
    Q_padd[0:-1,0:-1] = qr
    Q_padd[-1,:] = Q_padd[0,:]
    Q_padd[:,-1] = Q_padd[:,0]
    interp_f = RectBivariateSpline(y_padd, x_padd, Q_padd, kx=3, ky=3)
    qr_interp = interp_f.ev(np.ravel(Y_in), np.ravel(X_in))
    qr_interp = np.reshape(qr_interp, (qr.shape[0], qr.shape[1]))
    return fft2(qr_interp)


t_start = time.time()
for iTime in range(1, Nt_hf+1):
    t = iTime*dt
    # update momentum equation for u and v and h
    ur = np.real(ifft2(uk))
    vr = np.real(ifft2(vk))
    hr = np.real(ifft2(hk))
    u_xr = np.real(ifft2(1j * Kxx * uk))
    v_xr = np.real(ifft2(1j * Kxx * vk))
    u_yr = np.real(ifft2(1j * Kyy * uk))
    v_yr = np.real(ifft2(1j * Kyy * vk))
    Nhx = L * fft2(ur * (hr))
    Nhy = L * fft2(vr * (hr))
    Nu = L * fft2(ur * u_xr + vr * u_yr)
    Nv = L * fft2(ur * v_xr + vr * v_yr)
    hk = (-1j * Kxx * Nhx - 1j * Kyy * Nhy)*dt/2 + hk
    uk = Cc * (uk + (-Nu + vk/Ro - 1j * Kxx * hk/Fr**2)*dt/2)
    vk = Cc * (vk + (-Nv - uk/Ro - 1j * Kyy * hk/Fr**2)*dt/2)

    ur = np.real(ifft2(uk))
    vr = np.real(ifft2(vk))
    hr = np.real(ifft2(hk))
    u_xr = np.real(ifft2(1j * Kxx * uk))
    v_xr = np.real(ifft2(1j * Kxx * vk))
    u_yr = np.real(ifft2(1j * Kyy * uk))
    v_yr = np.real(ifft2(1j * Kyy * vk))
    Nhx = L * fft2(ur * (hr))
    Nhy = L * fft2(vr * (hr))
    Nu = L * fft2(ur * u_xr + vr * u_yr)
    Nv = L * fft2(ur * v_xr + vr * v_yr)
    hk = (-1j * Kxx * Nhx - 1j * Kyy * Nhy)*dt/2 + hk
    vk = Cc * (vk + (-Nv - uk/Ro - 1j * Kyy * hk/Fr**2)*dt/2)
    uk = Cc * (uk + (-Nu + vk/Ro - 1j * Kxx * hk/Fr**2)*dt/2)

    # ----------- Solving Lagrangian Mean PDEs ------------
    # find the instantaneous values of the quantity that we are going to
    # compute its Lag mean. (skip this step if it is already available from
    # the solution of governing equation)
    zr = np.real(ifft2(1j*Kxx*vk-1j*Kyy*uk))

    # =============>> 1) Solving the equation for \xi (displacement)
    Xix_xr = np.real(ifft2(1j * Kxx * Xixk))
    Xix_yr = np.real(ifft2(1j * Kyy * Xixk))
    Xiy_xr = np.real(ifft2(1j * Kxx * Xiyk))
    Xiy_yr = np.real(ifft2(1j * Kyy * Xiyk))
    # nonlinear terms in the \xi equation
    NXix = L * fft2(Xixr * Xix_xr + Xiyr * Xix_yr)
    NXiy = L * fft2(Xixr * Xiy_xr + Xiyr * Xiy_yr)
    # solving the \xi PDE using Explicit Euler
    Xixk = (((-NXix - Xixk)/t + uL_RHSk)*dt + Xixk)
    Xiyk = (((-NXiy - Xiyk)/t + vL_RHSk)*dt + Xiyk)
    # prepare fields for the next time step
    Xixr = np.real(ifft2(Xixk))
    Xiyr = np.real(ifft2(Xiyk))
    CurPosX = xx + Xixr
    CurPosY = yy + Xiyr
    X_in = CurPosX - np.floor(CurPosX/(2*np.pi))*(2*np.pi)
    Y_in = CurPosY - np.floor(CurPosY/(2*np.pi))*(2*np.pi)
    # padding for interpolation (side note 2) at the beginning of script)
    uL_RHSk = interp_fft(x_padd, y_padd, ur, X_in, Y_in)
    vL_RHSk = interp_fft(x_padd, y_padd, vr, X_in, Y_in)
    # uL_RHSk = interp_fft2(x, y, ur, X_in, Y_in)
    # vL_RHSk = interp_fft2(x, y, vr, X_in, Y_in)
    
    # =============>> 2) Solving the equation for \bar g 
    zL_RHSk = interp_fft(x_padd, y_padd, zr, X_in, Y_in)
    z_LM_xr = np.real(ifft2(1j * Kxx * z_LMk))
    z_LM_yr = np.real(ifft2(1j * Kyy * z_LMk))
    NzL = L * fft2(Xixr * z_LM_xr + Xiyr * z_LM_yr)
    z_LMk = ((-NzL/t + zL_RHSk)*dt + z_LMk)
    hL_RHSk = interp_fft(x_padd, y_padd, hr, X_in, Y_in)
    h_LM_xr = np.real(ifft2(1j * Kxx * h_LMk))
    h_LM_yr = np.real(ifft2(1j * Kyy * h_LMk))
    NhL = L * fft2(Xixr * h_LM_xr + Xiyr * h_LM_yr)
    h_LMk = ((-NhL/t + hL_RHSk)*dt + h_LMk)

    # adding fields to get the Eulerian mean
    z_EMr += zr
    h_EMr += hr

zr_middle = zr
hr_middle = hr

for iTime in range(Nt_hf+1, 2*Nt_hf+1):
    t = iTime*dt
    # update momentum equation for u and v and h
    ur = np.real(ifft2(uk))
    vr = np.real(ifft2(vk))
    hr = np.real(ifft2(hk))
    u_xr = np.real(ifft2(1j * Kxx * uk))
    v_xr = np.real(ifft2(1j * Kxx * vk))
    u_yr = np.real(ifft2(1j * Kyy * uk))
    v_yr = np.real(ifft2(1j * Kyy * vk))
    Nhx = L * fft2(ur * (hr))
    Nhy = L * fft2(vr * (hr))
    Nu = L * fft2(ur * u_xr + vr * u_yr)
    Nv = L * fft2(ur * v_xr + vr * v_yr)
    hk = (-1j * Kxx * Nhx - 1j * Kyy * Nhy)*dt/2 + hk
    uk = Cc * (uk + (-Nu + vk/Ro - 1j * Kxx * hk/Fr**2)*dt/2)
    vk = Cc * (vk + (-Nv - uk/Ro - 1j * Kyy * hk/Fr**2)*dt/2)

    ur = np.real(ifft2(uk))
    vr = np.real(ifft2(vk))
    hr = np.real(ifft2(hk))
    u_xr = np.real(ifft2(1j * Kxx * uk))
    v_xr = np.real(ifft2(1j * Kxx * vk))
    u_yr = np.real(ifft2(1j * Kyy * uk))
    v_yr = np.real(ifft2(1j * Kyy * vk))
    Nhx = L * fft2(ur * (hr))
    Nhy = L * fft2(vr * (hr))
    Nu = L * fft2(ur * u_xr + vr * u_yr)
    Nv = L * fft2(ur * v_xr + vr * v_yr)
    hk = (-1j * Kxx * Nhx - 1j * Kyy * Nhy)*dt/2 + hk
    vk = Cc * (vk + (-Nv - uk/Ro - 1j * Kyy * hk/Fr**2)*dt/2)
    uk = Cc * (uk + (-Nu + vk/Ro - 1j * Kxx * hk/Fr**2)*dt/2)


    # ----------- Solving Lagrangian Mean PDEs ------------
    # find the instantaneous values of the quantity that we are going to
    # compute its Lag mean. (skip this step if it is already available from
    # the solution of governing equation)
    zr = np.real(ifft2(1j*Kxx*vk-1j*Kyy*uk))

    # =============>> 1) Solving the equation for \xi (displacement)
    Xix_xr = np.real(ifft2(1j * Kxx * Xixk))
    Xix_yr = np.real(ifft2(1j * Kyy * Xixk))
    Xiy_xr = np.real(ifft2(1j * Kxx * Xiyk))
    Xiy_yr = np.real(ifft2(1j * Kyy * Xiyk))
    # nonlinear terms in the \xi equation
    NXix = L * fft2(Xixr * Xix_xr + Xiyr * Xix_yr)
    NXiy = L * fft2(Xixr * Xiy_xr + Xiyr * Xiy_yr)
    # solving the \xi PDE using Explicit Euler
    Xixk = (((-NXix - Xixk)/t + uL_RHSk)*dt + Xixk)
    Xiyk = (((-NXiy - Xiyk)/t + vL_RHSk)*dt + Xiyk)
    # prepare fields for the next time step
    Xixr = np.real(ifft2(Xixk))
    Xiyr = np.real(ifft2(Xiyk))
    CurPosX = xx + Xixr
    CurPosY = yy + Xiyr
    X_in = CurPosX - np.floor(CurPosX/(2*np.pi))*(2*np.pi)
    Y_in = CurPosY - np.floor(CurPosY/(2*np.pi))*(2*np.pi)
    # padding for interpolation (side note 2) at the beginning of script)
    uL_RHSk = interp_fft(x_padd, y_padd, ur, X_in, Y_in)
    vL_RHSk = interp_fft(x_padd, y_padd, vr, X_in, Y_in)
    
    # =============>> 2) Solving the equation for \bar g 
    zL_RHSk = interp_fft(x_padd, y_padd, zr, X_in, Y_in)
    z_LM_xr = np.real(ifft2(1j * Kxx * z_LMk))
    z_LM_yr = np.real(ifft2(1j * Kyy * z_LMk))
    NzL = L * fft2(Xixr * z_LM_xr + Xiyr * z_LM_yr)
    z_LMk = ((-NzL/t + zL_RHSk)*dt + z_LMk)
    hL_RHSk = interp_fft(x_padd, y_padd, hr, X_in, Y_in)
    h_LM_xr = np.real(ifft2(1j * Kxx * h_LMk))
    h_LM_yr = np.real(ifft2(1j * Kyy * h_LMk))
    NhL = L * fft2(Xixr * h_LM_xr + Xiyr * h_LM_yr)
    h_LMk = ((-NhL/t + hL_RHSk)*dt + h_LMk)


    # adding fields to get the Eulerian mean
    zr = np.real(ifft2(1j*Kxx*vk-1j*Kyy*uk))
    z_EMr += zr
    h_EMr += hr

z_EMr = z_EMr/iTime
h_EMr = h_EMr/iTime 
z_LMr = np.real(ifft2(z_LMk))/t
h_LMr = np.real(ifft2(h_LMk))/t

t_end = time.time()
print('computation time', t_end - t_start)


fig1, ax1 = plt.subplots(1,3, figsize = (13,4))

im1 = ax1[0].pcolor(xx,yy,zr_middle, cmap = 'RdBu_r', vmin=-1.8, vmax=2.7)
ax1[0].set_xlim(0,2*np.pi)
ax1[0].set_ylim(0,2*np.pi)
ax1[0].axis('equal')
ax1[0].axis('off')
divider = make_axes_locatable(ax1[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)
im2 = ax1[1].pcolor(xx,yy,z_LMr, cmap = 'RdBu_r', vmin=-.65, vmax=.65)
ax1[1].set_xlim(0,2*np.pi)
ax1[1].set_ylim(0,2*np.pi)
ax1[1].axis('equal')
ax1[1].axis('off')
divider = make_axes_locatable(ax1[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)
im3 = ax1[2].pcolor(xx,yy,z_EMr, cmap = 'RdBu_r', vmin=-.65, vmax=.65)
ax1[2].set_xlim(0,2*np.pi)
ax1[2].set_ylim(0,2*np.pi)
ax1[2].axis('equal')
ax1[2].axis('off')
divider = make_axes_locatable(ax1[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

fig2, ax2 = plt.subplots(1,3, figsize = (13,4))

ima1 = ax2[0].pcolor(xx,yy,(zr_middle+1/Ro)/hr_middle, cmap = 'RdBu_r', vmin=9, vmax=10.8)
ax2[0].set_xlim(0,2*np.pi)
ax2[0].set_ylim(0,2*np.pi)
ax2[0].axis('equal')
ax2[0].axis('off')
divider = make_axes_locatable(ax2[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ima1, cax=cax)
ima2 = ax2[1].pcolor(xx,yy,(z_LMr+1/Ro)/h_LMr, cmap = 'RdBu_r', vmin=9, vmax=10.8)
ax2[1].set_xlim(0,2*np.pi)
ax2[1].set_ylim(0,2*np.pi)
ax2[1].axis('equal')
ax2[1].axis('off')
divider = make_axes_locatable(ax2[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ima2, cax=cax)
ima3 = ax2[2].pcolor(xx,yy,(z_EMr+1/Ro)/h_EMr, cmap = 'RdBu_r', vmin=9, vmax=10.8)
ax2[2].set_xlim(0,2*np.pi)
ax2[2].set_ylim(0,2*np.pi)
ax2[2].axis('equal')
ax2[2].axis('off')
divider = make_axes_locatable(ax2[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ima3, cax=cax)

plt.show()


    

