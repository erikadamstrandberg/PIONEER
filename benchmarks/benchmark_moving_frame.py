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
project_folder_path = Path(__file__).parents[1]
PAS_PATH = Path(project_folder_path, 'PAS')
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

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.01)
    
    
    ax[0].imshow(I, cmap='jet', extent=extent)
    
    ax[0].set_title(r'Intensity')
    ax[0].set_xlabel(r'$x$ [mm]')
    ax[0].set_ylabel(r'$y$ [mm]')
    ''
    ax[1].imshow(phase, extent=extent)
    
    ax[1].set_title(r'Phase distribution')
    ax[1].set_xlabel(r'$x$ [mm]')
    ax[1].set_ylabel(r'$y$ [mm]')

    fig.suptitle(position)
        
def shift_frame(E2, offset_x, offset_y):
    E2 = np.roll(E2, offset_x, axis=1)
    if offset_x != 0:
        E2[:, 0:offset_x] = 0
        
    E2 = np.roll(E2, offset_y, axis=0)
    if offset_y != 0:
        E2[0:offset_y, :] = 0
        
    return E2
    
#%% Create beam
# Setup simulation window
Nx = 2**9
Ny = 2**9

Lx = 4e-3
Ly = 4e-3

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
omega1   = 80e-6

f_lins = 10e-3
T_lins = np.exp(-1j*k*R**2/(2*f_lins))

offset_x = 0
offset_y = 0

E1 = np.exp(-((X - offset_x)**2 + (Y - offset_y)**2)/omega1**2)*T_lins

E1_phase = np.angle(E1)
I1 = np.abs(E1)**2 
I1_norm = I1/np.max(I1)

x_mm = x*1e3
y_mm = y*1e3
extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()]

plot_starting_field = False
if plot_starting_field:
    plot_intensity_and_phase(I1_norm, E1_phase, x, y, 'Plane 1')
    
# Propagation to plane 2
L_prop = 22e-3
dz = 1e-3

L_prop_vect = np.arange(0, L_prop, dz)
L_prop_vect_mm = L_prop_vect*1e3
I_crossection_x = np.zeros(shape=(Ny, len(L_prop_vect)))
I_crossection_y = np.zeros(shape=(Nx, len(L_prop_vect)))


offset_x = 2
offset_y = 0


#%%
for i in range(len(L_prop_vect)):
    print('Propagating step ' + str(i) + '/' + str(len(L_prop_vect)))

    # Propagate to plane 2
    E2 = PAS(E1, dz, Nx, ax, lam0, n)
    # Intensity in plane 2                          
    I2 = np.abs(E2)**2
    I2 = I2/np.max(I2)
    
    I_crossection_x[:, i] = I2[:, int(Nx/2+1)]
    I_crossection_y[:, i] = I2[int(Nx/2+1), :]
    
    E1 = shift_frame(E2, offset_x, offset_y)
    

E2_phase = np.angle(E2)

plot_final_field = True
if plot_final_field:
    plot_intensity_and_phase(I2, E2_phase, x, y, 'Plane 2')  
    
#%%

data_save_path = Path(current_file_folder_path, 'data_moving_frame')
np.savetxt(Path(data_save_path, 'I_crossection_x.csv'), I_crossection_x, delimiter=',')
np.savetxt(Path(data_save_path, 'I_crossection_y.csv'), I_crossection_y, delimiter=',')
np.savetxt(Path(data_save_path, 'L_prop_vect_mm.csv'), L_prop_vect_mm, delimiter=',')
np.savetxt(Path(data_save_path, 'x_mm.csv'), x_mm, delimiter=',')
np.savetxt(Path(data_save_path, 'y_mm.csv'), y_mm, delimiter=',')


