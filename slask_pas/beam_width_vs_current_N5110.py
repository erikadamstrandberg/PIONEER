#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scipy import constants
c     = constants.speed_of_light
mu_0  = constants.mu_0
eps_0 = constants.epsilon_0

from pathlib import Path
project_folder = Path(__file__).parents[1]
refractive_index_folder = str(Path(project_folder,'REFRACTIVE_INDEX'))

import sys
if refractive_index_folder not in sys.path:
    sys.path.append(refractive_index_folder)
    
from refractive_index import n_AlGaAs_afromowitz as n_AlGaAs

#%%

def fft2c(x):
    '''
    2D Fourier transform with shift!
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))

def ifft2c(x):
    '''
    2D inverse Fourier transform with shift!
    '''
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x)))



def PAS(E1, L, N, a, lam0, n):
    '''
    Propagation of angular spectrum
    '''
    delta_k = 2*np.pi/(N*a)
    kx  = np.arange(-(N/2)*delta_k, (N/2)*delta_k, delta_k)
    ky  = kx
    KX, KY = np.meshgrid(kx,ky)
    
    k = 2*np.pi*n/lam0
    KZ = np.sqrt(k**2 - KX**2 - KY**2, dtype=complex)
    
    phase_prop = np.exp(1j*KZ*L)

    A = (a**2/(4*np.pi**2))*fft2c(E1)
    B = A*phase_prop
    E2 = (N*delta_k)**2*ifft2c(B)
    
    return E2

def M_border_TE(n1, n2, theta1, theta2):
    return (1/(2*n2*np.cos(theta2)))*np.array([[n2*np.cos(theta2) + n1*np.cos(theta1), n2*np.cos(theta2) - n1*np.cos(theta1)],
                                               [n2*np.cos(theta2) - n1*np.cos(theta1), n2*np.cos(theta2) + n1*np.cos(theta1)]])

# def M_border_TE(n1, n2, theta1, theta2):
#     return (1*np.cos(theta1)/(2*n2*np.cos(theta2)))*np.array([[n2*np.cos(theta2) + n1*np.cos(theta1), n2*np.cos(theta2) - n1*np.cos(theta1)],
#                                                [n2*np.cos(theta2) - n1*np.cos(theta1), n2*np.cos(theta2) + n1*np.cos(theta1)]])    
    
def M_border(n1, n2):
    return (1/(2*n2))*np.array([[n2 + n1, n2 - n1],
                                [n2 - n1, n2 + n1]])

def gauss_fit(x, norm_factor, mean, std):
    return norm_factor*np.exp(-2*((x - mean)/std)**2)


mA_measurement = np.array([0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                           2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2,
                           3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4. , 4.1, 4.2, 4.3, 4.4, 4.5,
                           4.6, 4.7, 4.8, 4.9, 5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8,
                           5.9, 6. , 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9])

width_measurement = np.array([341.2026769 , 350.97055278, 343.60718415, 343.35542346,
                              343.38689849, 346.88236182, 344.8775474 , 345.21343713,
                              344.97005986, 344.47353796, 345.96758322, 342.92398648,
                              346.73424528, 344.17535617, 347.86082561, 345.80405545,
                              349.19896967, 346.4141681 , 347.63890397, 347.03465034,
                              348.05855831, 347.30539126, 349.48803654, 348.67754952,
                              348.06223316, 346.80395303, 347.49617894, 347.78328674,
                              348.42352578, 348.63833182, 350.25427981, 349.15988935,
                              348.53491249, 349.37208959, 350.38410865, 350.55578159,
                              351.15045472, 351.27971025, 351.77284989, 350.74918999,
                              351.14560675, 351.51955211, 353.15567697, 353.32128313,
                              355.00246162, 356.67268916, 356.85120294, 357.18826423,
                              357.03995256, 357.24281455, 356.9725793 , 357.23070723,
                              357.5159382 , 357.79725811, 358.21509807, 359.13283763,
                              360.35214825, 360.13954389, 362.23488015, 362.83972878,
                              363.64373017, 364.06148923, 366.09241298])


plt.figure(1)
plt.plot(mA_measurement, width_measurement, 'black')

plt.xlabel(r'$I$ [mA]')
plt.ylabel(r'Beam width $1/e^2$ [$\mu$m]')
plt.title(r'Beam width vs. I, $d=1.98$ [$\mu$m]')

plt.grid(True)


#%% Testing the PAS algorithm 
#   Propagating guassian beams and plane waves

# Sampled points
N = 2**12
# Distance of plane 1
L1 = 2e-3
# Distance between sampled points in plane 1
a = L1/N

# Propagation distance for PAS to plane 2
L1 = 600e-6
L2 = 1e-3

# Wavelength
lam0 = 982e-9
# Propagation medium 
(n1, k1) = n_AlGaAs(lam0, 0)
n2 = 1.5

# Propagation constant
k1 = (2*np.pi*n1)/lam0 
k2 = (2*np.pi*n2)/lam0 
# Sampled points in plane 1
x = np.arange(-(N/2)*a, (N/2)*a, a)
y = x

# Meshgrids to create matrices with all X-Y coordinates 
X, Y = np.meshgrid(x, y)
# Matrix with distance to all points
R = np.sqrt(X**2 + Y**2)

# Waist of gaussian beam
omega1 = 0.6e-6
# Create gaussian beam in plane 1
E1 = np.exp(-R**2/omega1**2)

# Intensity in plane 1
I1 = np.abs(E1)**2 
I1_max = np.max(I1)

I1_plot = I1/I1_max

# Plot intensity in plane 1
# plt.figure(1)
# c = plt.imshow(I1_plot, extent =[x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()])

# plt.title(r'Intensity in plane 1')
# plt.xlabel(r'x $[$mm$]$')
# plt.ylabel(r'y $[$mm$]$')
# plt.colorbar(c)

# Propagate to plane 2
E2 = PAS(E1, L1, N, a, lam0, n1)

# E2_x_profile = E2[:, int(N/2+1)]
# I2_x_profile = np.abs(E2_x_profile)**2

# I2_x_profile_norm = I2_x_profile/np.max(I2_x_profile)

# parameters, covariance = curve_fit(gauss_fit, x, I2_x_profile_norm, p0=[1, 0, 0.001])
# norm_factor = parameters[0]
# mean = parameters[1]
# std = parameters[2]

x_um = x*1e6

# plt.figure(1)
# plt.plot(x_um, gauss_fit(x, norm_factor, mean, std))
# plt.plot(x_um, I2_x_profile_norm)

# Intensity in plane 2                          
I2 = np.abs(E2)**2

I2_max = np.max(I2)               
I2_plot = I2/I2_max

x_um = x*1e6
y_um = y*1e6

radius_MS = 200
theta = np.arange(0, 2*np.pi, 200)
x_MS = radius_MS*np.cos(theta)
y_MS = radius_MS*np.sin(theta)

plt.figure(2)
c = plt.imshow(I2_plot, extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()])
plt.scatter(0, 0, s=8000, facecolors='none', edgecolors='red', label=r'MS')
 

plt.title(r'Intensity profile at GaAs-interface')
plt.xlabel(r'x [$\mu$m]')
plt.ylabel(r'y [$\mu$m]')
plt.colorbar(c)


plt.xlim([-500, 500])
plt.ylim([-500, 500])


#%%
A = fft2c(E2)
border = M_border(n1, n2)
B = A*border[0,0]
E2 = ifft2c(B)

E3 = PAS(E2, L2, N, a, lam0, n2)

I3 = np.abs(E3)**2

I3_max = np.max(I3)               
I3_plot = I3/I3_max
                     
plt.figure(3)
c = plt.imshow(I3_plot, extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()])

plt.title(r'Intensity in plane 3')
plt.xlabel(r'x [$\mu$m]')
plt.ylabel(r'y [$\mu$m]')
plt.colorbar(c)

#%%

E3_x_profile = E3[:, int(N/2+1)]
I3_x_profile = np.abs(E3_x_profile)**2

I3_x_profile_norm = I3_x_profile/np.max(I3_x_profile)

plt.figure(4)
plt.plot(x_um, I3_x_profile_norm)

#%% Finding beam widths

def PAS_to_detector(L1, L2, omega1):
    # Sampled points
    N = 2**13
    # Distance of plane 1
    L = 10e-3
    # Distance between sampled points in plane 1
    a = L/N
    
    # Wavelength
    lam0 = 985e-9
    # Propagation medium 
    (n1, k1) = n_AlGaAs(lam0, 0)
    n2 = 1
    
    # Sampled points in plane 1
    x = np.arange(-(N/2)*a, (N/2)*a, a)
    y = x
    
    # Meshgrids to create matrices with all X-Y coordinates 
    X, Y = np.meshgrid(x, y)
    # Matrix with distance to all points
    R = np.sqrt(X**2 + Y**2)
    
    # Waist of gaussian beam
    # Create gaussian beam in plane 1
    E1 = np.exp(-R**2/omega1**2)
    
    # # Propagate to plane 2
    E2 = PAS(E1, L1, N, a, lam0, n1)
    
    E2_x_profile = E2[:, int(N/2+1)]
    I2_x_profile = np.abs(E2_x_profile)**2
    I2_x_profile_norm = I2_x_profile/np.max(I2_x_profile)
    parameters, covariance = curve_fit(gauss_fit, x, I2_x_profile_norm, p0=[1, 0, 0.0000001])
    
    std_interface = parameters[2]
    
    A = fft2c(E2)
    border = M_border(n1, n2)
    B = A*border[0,0]
    E2 = ifft2c(B)
    
    E3 = PAS(E2, L2, N, a, lam0, n2)
    
    E3_x_profile = E3[:, int(N/2+1)]
    I3_x_profile = np.abs(E3_x_profile)**2
    
    I3_x_profile_norm = I3_x_profile/np.max(I3_x_profile)
    
    return (I3_x_profile_norm, x, std_interface)


#%%
    
L1 = 600e-6
L2 = 1e-2
omega1 = 6e-6

(I3_x_profile_norm, x, std_interface) = PAS_to_detector(L1, L2, omega1)
print(std_interface*1e6)

parameters, covariance = curve_fit(gauss_fit, x, I3_x_profile_norm, p0=[1, 0, 0.00001])
norm_factor = parameters[0]
mean = parameters[1]
std = parameters[2]


x_um = x*1e6

plt.figure(1)
plt.plot(x_um, gauss_fit(x, norm_factor, mean, std))
plt.plot(x_um, I3_x_profile_norm)



