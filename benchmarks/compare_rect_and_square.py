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

#%% Recangular data

data_save_path = Path(current_file_folder_path, 'data_square_rect')

I_crossection_x_square = np.loadtxt(Path(data_save_path, 'I_crossection_x.csv'), delimiter=',')
I_crossection_y_square = np.loadtxt(Path(data_save_path, 'I_crossection_y.csv'), delimiter=',')
L_prop_vect_mm_square = np.loadtxt(Path(data_save_path, 'L_prop_vect_mm.csv'), delimiter=',')
x_mm_square = np.loadtxt(Path(data_save_path, 'x_mm.csv'), delimiter=',')
y_mm_square = np.loadtxt(Path(data_save_path, 'y_mm.csv'), delimiter=',')

# #%%
# extent = [L_prop_vect_mm_square.min(), L_prop_vect_mm_square.max(), x_mm_square.min(), x_mm_square.max()]
# plt.imshow(I_crossection_y_square, extent=extent)

# #%%
# extent = [L_prop_vect_mm_square.min(), L_prop_vect_mm_square.max(), y_mm_square.min(), y_mm_square.max()]
# plt.imshow(I_crossection_x_square, extent=extent)

I_crossection_x_rect = np.loadtxt(Path(data_save_path, 'I_crossection_x_rect.csv'), delimiter=',')
I_crossection_y_rect = np.loadtxt(Path(data_save_path, 'I_crossection_y_rect.csv'), delimiter=',')
L_prop_vect_mm_rect = np.loadtxt(Path(data_save_path, 'L_prop_vect_mm_rect.csv'), delimiter=',')
x_mm_rect = np.loadtxt(Path(data_save_path, 'x_mm_rect.csv'), delimiter=',')
y_mm_rect = np.loadtxt(Path(data_save_path, 'y_mm_rect.csv'), delimiter=',')

# #%%
# extent = [L_prop_vect_mm_rect.min(), L_prop_vect_mm_rect.max(), x_mm_rect.min(), x_mm_rect.max()]
# plt.imshow(I_crossection_x_rect, extent=extent)

# #%%
# extent = [L_prop_vect_mm_rect.min(), L_prop_vect_mm_rect.max(), y_mm_rect.min(), y_mm_rect.max()]
# plt.imshow(I_crossection_y_rect, extent=extent)

#%%

last_prop_crossection_y_norm_square = I_crossection_y_square[:,-1]/np.max(I_crossection_y_square[:,-1])
last_prop_crossection_y_norm_rect = I_crossection_y_rect[:,-1]/np.max(I_crossection_y_rect[:,-1])

last_prop_crossection_x_norm_square = I_crossection_x_square[:,-1]/np.max(I_crossection_x_square[:,-1])
last_prop_crossection_x_norm_rect = I_crossection_x_rect[:,-1]/np.max(I_crossection_x_rect[:,-1])


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.01)
# fig.tight_layout()


ax[0].plot(x_mm_rect, last_prop_crossection_y_norm_rect, 'red', linewidth=2)
ax[0].plot(x_mm_square, last_prop_crossection_y_norm_square, '--', color='black', linewidth=2)

ax[0].set_title(r'Intensity y-crossections')
ax[0].set_xlabel(r'$x$ [mm]')
ax[0].set_ylabel(r'I [-]')
ax[0].grid(True)

ax[1].plot(y_mm_rect, last_prop_crossection_x_norm_rect, 'red', linewidth=2)
ax[1].plot(y_mm_square, last_prop_crossection_x_norm_square, '--', color='black', linewidth=2)

ax[1].set_title(r'Intensity x-crossections')
ax[1].set_xlabel(r'$y$ [mm]')
ax[1].set_ylabel(r'I [-]')
ax[1].grid(True)

fig.suptitle(r'Comparing x-y crossections')


