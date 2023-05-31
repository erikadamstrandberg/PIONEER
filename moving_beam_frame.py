#%%

## Standard imports for math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']

## Import modules
import sys
from pathlib import Path
current_file_folder_path = Path(__file__).parents[0]
PAS_PATH = Path(current_file_folder_path, 'PAS')
PAS_PATH_string = str(PAS_PATH)
if PAS_PATH_string not in sys.path:
    sys.path.append(PAS_PATH_string)

from pas_functions import PAS, fft2c, ifft2c

from scipy import constants
c     = constants.speed_of_light
mu_0  = constants.mu_0
eps_0 = constants.epsilon_0

def plot_intensity_and_phase(I, phase, x, y, position):    
    x_mm = x*1e3
    y_mm = y*1e3
    extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()]

    if plot_starting_field:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.01)
        # fig.tight_layout()
        
        
        ax[0].imshow(I, cmap='jet', extent=extent)
        
        ax[0].set_title(r'Intensity')
        ax[0].set_xlabel(r'$x$ [mm]')
        ax[0].set_ylabel(r'$y$ [mm]')
        
        ax[1].imshow(phase, extent=extent)
        
        ax[1].set_title(r'Phase distribution')
        ax[1].set_xlabel(r'$x$ [mm]')
        ax[1].set_ylabel(r'$y$ [mm]')

        fig.suptitle(position)

    
#%% Create beam
# Setup simulation window
Nx = 2**10
Ny = 2**10

Lx = 3e-3
Ly = 3e-3

ax = Lx/Nx
ay = Ly/Ny

x = np.arange(-(Nx/2)*ax, (Nx/2)*ax, ax)
y = np.arange(-(Ny/2)*ay, (Ny/2)*ay, ay)

X, Y = np.meshgrid(x, y)
R    = np.sqrt(X**2 + Y**2)

# Setup beam
lam0     = 980e-9
n        = 1
k        = (2*np.pi*n)/lam0 
omega1   = 20e-6

# f_lins = 600e-3
# T_lins = np.exp(-1j*k*R**2/(2*f_lins))

E1 = np.exp(-R**2/omega1**2)

E1_phase = np.angle(E1)
I1 = np.abs(E1)**2 
I1_norm = I1/np.max(I1)

x_mm = x*1e3
y_mm = y*1e3
extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()]

plot_starting_field = False
if plot_starting_field:
    plot_intensity_and_phase(I1_norm, E1_phase, x, y, 'Plane 1')
    
# Propagation 
L_prop = 10000e-6
dz = 300e-6

L_prop_vect = np.arange(0, L_prop, dz)
I_crossection_x = np.zeros(shape=(Ny, len(L_prop_vect)))
I_crossection_y = np.zeros(shape=(Nx, len(L_prop_vect)))

# x_mm = x*1e3
# y_mm = y*1e3

for i in range(len(L_prop_vect)):

    print('Propagating step ' + str(i) + '/' + str(len(L_prop_vect)))

    # Propagate to plane 2
    E2 = PAS(E1, L_prop_vect[i], Nx, ax, lam0, n)
    # Intensity in plane 2                          
    I2 = np.abs(E2)**2
    I2 = I2/np.max(I2)
    
    I_crossection_y[:, i] = I2[:, int(Nx/2+1)]
    
E_metasurface_interface = E2

extent = [L_prop_vect.min()*1e3, L_prop_vect.max()*1e3, y_mm.min(), y_mm.max()]
plt.imshow(I_crossection_y, extent=extent)

    
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



