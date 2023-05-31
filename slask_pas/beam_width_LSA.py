#%%
import numpy as np

import matplotlib.pyplot as plt
# import tikzplotlib as tikz

import scipy.io

from scipy import constants
c     = constants.speed_of_light
mu_0  = constants.mu_0
eps_0 = constants.epsilon_0

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
    
#%% Testing the PAS algorithm 
#   Propagating guassian beams and plane waves

# Sampled points
N = 2**12
# Distance of plane 1
L1 = 3e-3
# Distance between sampled points in plane 1
a = L1/N

# Propagation distance for PAS to plane 2
L = 600e-6
dz = 50e-6

# Wavelength
lam0 = 980e-9
# Propagation medium 
n = 3.5
# Propagation constant
k = (2*np.pi*n)/lam0 
# Sampled points in plane 1
x = np.arange(-(N/2)*a, (N/2)*a, a)
y = x

# Meshgrids to create matrices with all X-Y coordinates 
X, Y = np.meshgrid(x, y)
# Matrix with distance to all points
R = np.sqrt(X**2 + Y**2)

# Waist of gaussian beam
omega1 = 1.01e-6
# Create gaussian beam in plane 1
E1_gauss = np.exp(-R**2/omega1**2)
E1_gauss_lens = E1_gauss

# Choose what field to propagate
E1 = E1_gauss_lens

# Intensity in plane 1
I1 = np.abs(E1)**2 
I1_max = np.max(I1)

I1_plot = I1/I1_max

x_mm = x*1e3
y_mm = y*1e3

L_vekt_1 = np.arange(0, L, dz)
I_crossection_1 = np.zeros(shape=(N, len(L_vekt_1)))

for i in range(len(L_vekt_1)):
    print(i)

    # Propagate to plane 2
    E2 = PAS(E1, L_vekt_1[i], N, a, lam0, n)
    # Intensity in plane 2                          
    I2 = np.abs(E2)**2
    I2 = I2/np.max(I2)
    
    I_crossection_1[:, i] = I2[:, int(N/2+1)]
    
E_metasurface_interface = E2

extent = [L_vekt_1.min()*1e3, L_vekt_1.max()*1e3, y_mm.min(), y_mm.max()]
plt.imshow(I_crossection_1, extent=extent)

    
#%%

first_half = I_crossection_1[0:int(N/2), -1]
second_half = I_crossection_1[int(N/2):, -1]

first_index = np.argmin(np.abs(first_half - np.exp(-2)))
second_index = np.argmin(np.abs(second_half - np.exp(-2))) + len(first_half)

x_first = np.array([y_mm[first_index], y_mm[first_index]])
x_second = np.array([y_mm[second_index], y_mm[second_index]])
y = np.array([0, 1])
# plt.plot(y_mm, np.concatenate((first_half, second_half)))
plt.plot(y_mm, I_crossection_1[:, -1])
plt.plot(x_first, y)
plt.plot(x_second, y)

#%%

beam_width_mm = y_mm[second_index] - y_mm[first_index]
beam_width_um = beam_width_mm*1e3
print(beam_width_um)

#%%
n = 1
# Propagation constant
k = (2*np.pi*n)/lam0 

# Propagation distance for PAS to plane 2
L = 22e-3
dz = 0.5e-3

# Focal length of lens
f_lins = 100e-6
# T_lins = np.exp(-1j*2*np.pi*(np.sqrt(R**2 + f_lins**2) - f_lins)/lam0)
# T_lins = np.exp(-1j*k*R**2/(2*f_lins))
T_lins = np.exp(-1j*np.angle(E_metasurface_interface))


# plt.imshow(np.imag(T_lins_perfect - T_lins))

# #%%

# Choose what field to propagate
E1 = E_metasurface_interface*T_lins
# plt.imshow(np.angle(T_lins))

# plt.imshow(np.angle(E1))
L_vekt_2 = np.arange(0, L, dz)
I_crossection_2 = np.zeros(shape=(N, len(L_vekt_2)))


#%%
for i in range(len(L_vekt_2)):
    print(i)

    # Propagate to plane 2
    E2 = PAS(E1, L_vekt_2[i], N, a, lam0, n)
    # Intensity in plane 2                          
    I2 = np.abs(E2)**2
    I2 = I2/np.max(I2)
    
    I_crossection_2[:, i] = I2[:, int(N/2+1)]
    

extent = [L_vekt_2.min()*1e3, L_vekt_2.max()*1e3, y_mm.min(), y_mm.max()]
plt.imshow(I_crossection_2, extent=extent)

#%%

L_vekt = np.concatenate((L_vekt_1, L_vekt_2 + L_vekt_1.max() + dz))
I_crossection = np.concatenate((I_crossection_1, I_crossection_2),axis=1)

extent = [L_vekt.min()*1e3, L_vekt.max()*1e3, y_mm.min(), y_mm.max()]
plt.imshow(I_crossection, extent=extent, cmap='jet')

x_substrate = np.array([0.6, 0.6])
y_substrate = np.array([y_mm.min(), y_mm.max()])
plt.plot(x_substrate, y_substrate, 'r')

plt.title('Propagation from aperture to 22 mm')
plt.xlabel('z [mm]')
plt.ylabel('y [mm]')

#%%

first_half = I_crossection_2[0:int(N/2), -1]
second_half = I_crossection_2[int(N/2):, -1]

first_index = np.argmin(np.abs(first_half - np.exp(-2)))
second_index = np.argmin(np.abs(second_half - np.exp(-2))) + len(first_half)

x_first = np.array([y_mm[first_index], y_mm[first_index]])
x_second = np.array([y_mm[second_index], y_mm[second_index]])
y = np.array([0, 1])
# plt.plot(y_mm, np.concatenate((first_half, second_half)))
plt.plot(y_mm, I_crossection_2[:, -1])
plt.plot(x_first, y)
plt.plot(x_second, y)

#%%
beam_width_mm = y_mm[second_index] - y_mm[first_index]
beam_width_um = beam_width_mm*1e3
print(beam_width_um)



