%% ============= About this code ==========================================
% This code solves Shallow-Water (SW) using a pseudo-spectral method 
% The equations are solved for variables:
% (u,v)=(vel. components) and h = displacement of free surface
% The SW equations are solved in the dimensionless form
% In tandem with the SW equations, the Lagrangian means are calculated
% using the strategy 2 of Kafiabad & Vanneste (2022)
% in the current version the Lag mean of vorticity and height field are 
% computed but the code can be easily used to compute any Lag means
% Developed by Hossein Kafiabad 2022 (kafiabad@gmail.com) 
% *************************************************************************
% Main variables: 
% uk, vk: velocities in Fourier space -- ur, vr: velocities in real space
% hk: height in Fourier space -- hr: height in real space
% zk: vorticity in Fourier space -- zr: vorticity in real space
% *************************************************************************
% Side notes:
% 1) In this code we want to store the instantaneous fields at the mid of 
% averaging interval. Hence, we compute the partial means in the first half
% of interval, then store the instantaneous values (e.g. for vorticity),
% and finally continue with the second half of the interval 
% (this is more efficient than leave an if statement inside one time loop)
% 2) For interpolation we padd the fields by adding the first row/column to
% the matrices after the last row/column. This is necessary for
% interpolation. For example, consider the periodic domain of [0, 10] which 
% is discretised by 10 points. The function at x = 0 and x= 10 is the same. 
% So in the pseudo-spectral code we use the disretised points 0,1,2 ..,9 as
% the information at t=10 is redundant. However, if we need to interpolate
% and the desired point is between 9 and 10 (like x=9.5), the interpolation
% function cannot handle it as 9.5 does not exist between two grid points.
% The padding throughout this script will take care of that.

clear
close all

%% ============= Parameters ===============================================
Re   = 3.84e3;     % Reynolds Number
Ro   = 0.1;        % Rossby Number
Fr   = 0.5;        % Froud Number
NX   = 128;        % number of discretised points in x- direction
NY   = 128;        % number of discretised points in y- direction
Lx   = 2*pi;       % Length of domain in x-direction
Ly   = 2*pi;       % Length of domain in y-direction
dt   = 2.5e-4;     % time step
Tend = 2.2;        % final time
Tave = Tend/2;     % means are assigned to Tave (mid of averaging interval)
Nt_half = floor(Tave/dt);


%% ============= Spatial variables + wavenumbers ==========================
% +++++++++++++++++++++++++++++++++++++++++++
% 1) in Real Space
% grid spacing in real space
dx=2*pi/NX;
dy=2*pi/NY;
% grid points in real space
[xx,yy]=meshgrid((0:(NX-1))*dx,(0:(NY-1))*dy);
% padding xx and yy only for interpolation (according to periodic BC)
% See the side note 2) at the beginning of the script
[xx_padded, yy_padded] = meshgrid((0:NX)*dx,(0:NY)*dy);
% +++++++++++++++++++++++++++++++++++++++++++
% 2) in Fourier spcae
kx = [0:(NX/2-1) (-NX/2):-1];
ky = kx';
[kxx,kyy]=meshgrid(kx,ky);
% k2 is used for dissipation term i.e. -nu*k^2 
% (we dont have issue with k2 =0)
% k2poisson is for solving poisson equation
k2=kxx.^2+kyy.^2;
k2poisson=kxx.^2+kyy.^2;
k2poisson(k2poisson==0)=1;

% de-aliasing mask: forces the nonlinear terms for kx,ky>2/3 to be zero
% depending on the problem and the type of dissipation can be relaxed ...
L=ones(size(k2));
L(abs(kxx)>(max(kx)*2/3))=0;
L(abs(kyy)>(max(ky)*2/3))=0;


%% ============= Initial condition ========================================
% +++++++++++++++++++++++++++++++++++++++++++
% Define the initial condition in real space

% ***** 1) Geostrophic flow: *****
% consist of a turbulent distribution of vortices that
% is a solution of 2D Navier Stokes & stored in'uvk_2Dturbulence_256.mat'
load('uvr_2Dturbulence_256.mat')
ur = ur(1:2:end,1:2:end);
vr = vr(1:2:end,1:2:end);
uk=fft2(ur);
vk=fft2(vr);
zk = 1i*kxx.*vk-1i*kyy.*uk;
% Find the associated height field such that it would be in geostrophic
% balance with the velocity fields
hk = -(Ro/Fr^2)*zk./k2poisson;
hr = 1 + real(ifft2(hk));

% ***** 2) Poincare wave: *****
Uw = 1.8;
kw = 1;
% find the wave frequency
omeg = sqrt(Ro^-2+Fr^-2*kw^2);
ur_wave =  -Uw*cos(kw*xx);
vr_wave =  -Uw/omeg/Ro*sin(kw*xx);
hr_wave =  -Uw/omeg*cos(kw*xx);

% Adding the wave to the flow
ur = ur + ur_wave;
vr = vr + vr_wave;
hr = hr + hr_wave;
hk=fft2(hr);
uk=fft2(ur);
vk=fft2(vr);

% the initial value for LM quantities that are going to be updated in time
zk = 1i*kxx.*vk-1i*kyy.*uk;
zr=real(ifft2(zk));
z_EMr = zr;
h_EMr = hr;

% initial values for Lag Mean PDEs
z_LMk = zeros(size(kxx));
h_LMk = zeros(size(kxx));
Xixr = zeros(size(xx));
Xiyr = zeros(size(yy));
Xixk = zeros(size(kxx));
Xiyk = zeros(size(kyy));
X_prt_in = xx;
Y_prt_in = yy;
uL_RHSk = ur;
vL_RHSk = vr;
zL_RHSk = zr;
hL_RHSk = hr;

%% Calculate the coefficients out of the loop (for numerical efficiency)
% Aa and Bb can be used for Crank-Nicholson scheme and Cc for exact sol.
% Calculate these values out of time-loop for efficiency 
% Aa = (1/dt-nu.*k2/2);
% Bb = 1./(1/dt+nu.*k2/2);
Cc = exp(-dt/Re.*k2);

%% ============= !! Time Steps !! =========================================


for iTime=1:Nt_half
    t = iTime*dt;
    
    ur = real(ifft2(uk));
    vr = real(ifft2(vk));
    hr = real(ifft2(hk));
    u_xr = real(ifft2(1i*kxx.*uk));
    v_xr = real(ifft2(1i*kxx.*vk));
    u_yr = real(ifft2(1i*kyy.*uk));
    v_yr = real(ifft2(1i*kyy.*vk));
    Nhx = L.*fft2(ur.*hr);
    Nhy = L.*fft2(vr.*hr);
    Nu = L.*fft2(ur.*u_xr+vr.*u_yr);
    Nv = L.*fft2(ur.*v_xr+vr.*v_yr);
    hk = (-1i*kxx.*Nhx-1i*kyy.*Nhy)*dt/2 + hk;  
    uk = ((-Nu + vk/Ro -1i*kxx.*hk/Fr^2)*dt/2 + uk).*Cc;
    vk = ((-Nv - uk/Ro -1i*kyy.*hk/Fr^2)*dt/2 + vk).*Cc;
    
    ur = real(ifft2(uk));
    vr = real(ifft2(vk));
    hr = real(ifft2(hk));
    u_xr = real(ifft2(1i*kxx.*uk));
    v_xr = real(ifft2(1i*kxx.*vk));
    u_yr = real(ifft2(1i*kyy.*uk));
    v_yr = real(ifft2(1i*kyy.*vk));
    Nhx = L.*fft2(ur.*hr);
    Nhy = L.*fft2(vr.*hr);
    Nu = L.*fft2(ur.*u_xr+vr.*u_yr);
    Nv = L.*fft2(ur.*v_xr+vr.*v_yr); 
    hk = (-1i*kxx.*Nhx-1i*kyy.*Nhy)*dt/2 + hk; 
    vk = ((-Nv - uk/Ro -1i*kyy.*hk/Fr^2)*dt/2 + vk).*Cc;
    uk = ((-Nu + vk/Ro -1i*kxx.*hk/Fr^2)*dt/2 + uk).*Cc;
    
    
    % ----------- Solving Lagrangian Mean PDEs ------------
    % find the instantaneous values of the quantity that we are going to
    % compute its Lag mean. (skip this step if it is already available from
    % the solution of governing equation)
    zk = 1i*kxx.*vk-1i*kyy.*uk;
    zr=real(ifft2(zk));
    
    % =============>> 1) Solving the equation for \xi (displacement)
    Xix_xr = real(ifft2(1i*kxx.*Xixk));
    Xix_yr = real(ifft2(1i*kyy.*Xixk));
    Xiy_xr = real(ifft2(1i*kxx.*Xiyk));
    Xiy_yr = real(ifft2(1i*kyy.*Xiyk));
    % nonlinear terms in the \xi equation
    NXix = L.*fft2(Xixr.*Xix_xr+Xiyr.*Xix_yr);
    NXiy = L.*fft2(Xixr.*Xiy_xr+Xiyr.*Xiy_yr);
    % solving the \xi PDE using Explicit Euler
    Xixk = (((-NXix - Xixk)/t + uL_RHSk)*dt + Xixk);
    Xiyk = (((-NXiy - Xiyk)/t + vL_RHSk)*dt + Xiyk);
    
    % prepare fields for the next time step
    Xixr = real(ifft2(Xixk));
    Xiyr = real(ifft2(Xiyk));
    CurPosX = xx+Xixr;
    CurPosY = yy+Xiyr;
    X_prt_in = CurPosX - floor(CurPosX/(2*pi))*(2*pi);
    Y_prt_in = CurPosY - floor(CurPosY/(2*pi))*(2*pi);
    % padding for interpolation (side note 2) at the beginning of script)
    Q1 = [ur;ur(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    uL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear'));
    Q1 = [vr;vr(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    vL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear'));
    
    % =============>> 2) Solving the equation for \bar g 
    %                    (partial lag mean in terms of mean position)
    Q1 = [zr;zr(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    zL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear')); 
    Q1 = [hr;hr(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    hL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear')); 
    z_LM_xr = real(ifft2(1i*kxx.*z_LMk));
    z_LM_yr = real(ifft2(1i*kyy.*z_LMk));
    NzL = L.*fft2(Xixr.*z_LM_xr+Xiyr.*z_LM_yr);
    z_LMk = ((-NzL/t + zL_RHSk)*dt + z_LMk);
    h_LM_xr = real(ifft2(1i*kxx.*h_LMk));
    h_LM_yr = real(ifft2(1i*kyy.*h_LMk));
    NhL = L.*fft2(Xixr.*h_LM_xr+Xiyr.*h_LM_yr);
    h_LMk = ((-NhL/t + hL_RHSk)*dt + h_LMk);
    
    % Add the instant. values to compute Eulerian mean later
    z_EMr = z_EMr + zr;
    h_EMr = h_EMr + hr;
end

zr_middle = zr;
hr_middle = hr;

for iTime=Nt_half+1:2*Nt_half
    t = iTime*dt;
    
    ur = real(ifft2(uk));
    vr = real(ifft2(vk));
    hr = real(ifft2(hk));
    u_xr = real(ifft2(1i*kxx.*uk));
    v_xr = real(ifft2(1i*kxx.*vk));
    u_yr = real(ifft2(1i*kyy.*uk));
    v_yr = real(ifft2(1i*kyy.*vk));
    Nhx = L.*fft2(ur.*hr);
    Nhy = L.*fft2(vr.*hr);
    Nu = L.*fft2(ur.*u_xr+vr.*u_yr);
    Nv = L.*fft2(ur.*v_xr+vr.*v_yr);
    hk = (-1i*kxx.*Nhx-1i*kyy.*Nhy)*dt/2 + hk;  
    uk = ((-Nu + vk/Ro -1i*kxx.*hk/Fr^2)*dt/2 + uk).*Cc;
    vk = ((-Nv - uk/Ro -1i*kyy.*hk/Fr^2)*dt/2 + vk).*Cc;
    
    ur = real(ifft2(uk));
    vr = real(ifft2(vk));
    hr = real(ifft2(hk));
    u_xr = real(ifft2(1i*kxx.*uk));
    v_xr = real(ifft2(1i*kxx.*vk));
    u_yr = real(ifft2(1i*kyy.*uk));
    v_yr = real(ifft2(1i*kyy.*vk));
    Nhx = L.*fft2(ur.*hr);
    Nhy = L.*fft2(vr.*hr);
    Nu = L.*fft2(ur.*u_xr+vr.*u_yr);
    Nv = L.*fft2(ur.*v_xr+vr.*v_yr); 
    hk = (-1i*kxx.*Nhx-1i*kyy.*Nhy)*dt/2 + hk; 
    vk = ((-Nv - uk/Ro -1i*kyy.*hk/Fr^2)*dt/2 + vk).*Cc;
    uk = ((-Nu + vk/Ro -1i*kxx.*hk/Fr^2)*dt/2 + uk).*Cc;
    
    
    % ----------- Solving Lagrangian Mean PDEs ------------
    % find the instantaneous values of the quantity that we are going to
    % compute its Lag mean. (skip this step if it is already available from
    % the solution of governing equation)
    zk = 1i*kxx.*vk-1i*kyy.*uk;
    zr=real(ifft2(zk));
    
    % =============>> 1) Solving the equation for \xi (displacement)
    Xix_xr = real(ifft2(1i*kxx.*Xixk));
    Xix_yr = real(ifft2(1i*kyy.*Xixk));
    Xiy_xr = real(ifft2(1i*kxx.*Xiyk));
    Xiy_yr = real(ifft2(1i*kyy.*Xiyk));
    % nonlinear terms in the \xi equation
    NXix = L.*fft2(Xixr.*Xix_xr+Xiyr.*Xix_yr);
    NXiy = L.*fft2(Xixr.*Xiy_xr+Xiyr.*Xiy_yr);
    % solving the \xi PDE using Explicit Euler
    Xixk = (((-NXix - Xixk)/t + uL_RHSk)*dt + Xixk);
    Xiyk = (((-NXiy - Xiyk)/t + vL_RHSk)*dt + Xiyk);
    
    % prepare fields for the next time step
    Xixr = real(ifft2(Xixk));
    Xiyr = real(ifft2(Xiyk));
    CurPosX = xx+Xixr;
    CurPosY = yy+Xiyr;
    X_prt_in = CurPosX - floor(CurPosX/(2*pi))*(2*pi);
    Y_prt_in = CurPosY - floor(CurPosY/(2*pi))*(2*pi);
    % padding for interpolation (side note 2) at the beginning of script)
    Q1 = [ur;ur(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    uL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear'));
    Q1 = [vr;vr(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    vL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear'));
    
    % =============>> 2) Solving the equation for \bar g 
    %                    (partial lag mean in terms of mean position)
    Q1 = [zr;zr(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    zL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear')); 
    Q1 = [hr;hr(1,:)];
    Qpadded = [Q1,Q1(:,1)];
    hL_RHSk = fft2(interp2(xx_padded,yy_padded,Qpadded,X_prt_in,Y_prt_in,'linear')); 
    z_LM_xr = real(ifft2(1i*kxx.*z_LMk));
    z_LM_yr = real(ifft2(1i*kyy.*z_LMk));
    NzL = L.*fft2(Xixr.*z_LM_xr+Xiyr.*z_LM_yr);
    z_LMk = ((-NzL/t + zL_RHSk)*dt + z_LMk);
    h_LM_xr = real(ifft2(1i*kxx.*h_LMk));
    h_LM_yr = real(ifft2(1i*kyy.*h_LMk));
    NhL = L.*fft2(Xixr.*h_LM_xr+Xiyr.*h_LM_yr);
    h_LMk = ((-NhL/t + hL_RHSk)*dt + h_LMk);
    
    % Add the instant. values to compute Eulerian mean later
    z_EMr = z_EMr + zr;
    h_EMr = h_EMr + hr;
end


z_EMr = z_EMr/iTime;
h_EMr = h_EMr/iTime;
z_LMr = real(ifft2(z_LMk))/t;
h_LMr = real(ifft2(h_LMk))/t;


%% ==================== plotting ====================  
% --------- vorticity field: instan., Lag mean, Eul mean ----------    
scnsize = get(0,'ScreenSize');
set(gcf,'Position',[20 20 .85*scnsize(3) .45*scnsize(4)]);
p = panel();
p.pack('h',{1.1/3.2 1/3.2 1.1/3.2})
p.margin = 12;
p.margintop = 12;
p.marginright = 12;
p(1).select();
title('(a)','FontSize',18)
% pcolor(xx,yy,(zr_middle+cor)./(hr_middle));caxis([3.9 6.2])
pcolor(xx,yy,zr_middle);caxis([-1.8 2.7])
shading interp;colormap(gca,othercolor('BuDRd_12'));colorbar;box on 
axis square
xlim([dx 2*pi]);ylim([dy 2*pi]);
% caxis([-.06 0.06])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[],'FontSize',18)
p(2).select();
title('(b)','FontSize',18)
%pcolor(xx,yy,(z_EMr+cor)./(h_EMr));caxis([3.9 6.2])
pcolor(xx,yy,z_LMr);caxis([-.65 0.65])
shading interp;colormap(gca,othercolor('BuDRd_12'));box on 
axis square
xlim([dx 2*pi]);ylim([dy 2*pi]);
% caxis([-.06 0.06])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[],'FontSize',18)
p(3).select();
title('(c)','FontSize',18)
%pcolor(xx,yy,(z_LMr+cor)/(h_LMr));caxis([3.9 6.2])
pcolor(xx,yy,z_EMr);caxis([-.65 0.65])
shading interp;colormap(gca,othercolor('BuDRd_12'));colorbar;hold on 
axis square
xlim([dx 2*pi]);ylim([dy 2*pi]);
%caxis([-.06 0.06])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[],'FontSize',18)



% --------- PV: instan., Lag mean, Eul mean ----------   
figure(2)
scnsize = get(0,'ScreenSize');
set(gcf,'Position',[20 20 .85*scnsize(3) .45*scnsize(4)]);
p = panel();
p.pack('h',{1/3.14 1/3.14 1.14/3.14})
p.margin = 12;
p.margintop = 12;
p.marginright = 12;
p(1).select();
title('(a)','FontSize',18)
pcolor(xx,yy,(zr_middle+1/Ro)./(hr_middle));caxis([9 10.8])
shading interp;colormap(gca,othercolor('BuDRd_12'));
axis square
xlim([dx 2*pi]);ylim([dy 2*pi]);
% caxis([-.06 0.06])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[],'FontSize',18)
p(2).select();
title('(b)','FontSize',18)
pcolor(xx,yy,(z_LMr+1/Ro)./(h_LMr));caxis([9 10.8])
shading interp;colormap(gca,othercolor('BuDRd_12'));box on 
axis square
xlim([dx 2*pi]);ylim([dy 2*pi]);
% caxis([-.06 0.06])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[],'FontSize',18)
p(3).select();
title('(c)','FontSize',18)
pcolor(xx,yy,(z_EMr+1/Ro)./(h_EMr));caxis([9 10.8])
shading interp;colormap(gca,othercolor('BuDRd_12'));colorbar;hold on 
axis square
xlim([dx 2*pi]);ylim([dy 2*pi]);
%caxis([-.06 0.06])
set(gca,'xtick',[],'xticklabel',[],'ytick',[],'yticklabel',[],'FontSize',18) 


