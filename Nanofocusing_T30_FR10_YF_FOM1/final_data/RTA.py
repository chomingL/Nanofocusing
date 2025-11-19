#!/usr/bin/env python
# coding: utf-8

# In[1]:


import meep as mp
import meep.adjoint as mpa
from meep import Animate2D
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from meep.materials import Ag
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import matplotlib.colors as mcolors

mp.verbosity(1)
TiO2 = mp.Medium(index=2.6)
SiO2 = mp.Medium(index=1.44)
Si = mp.Medium(index=3.4)
Air = mp.Medium(index=1)


# In[2]:
os.makedirs('TRAN',exist_ok=True) # Create Folder

##########################################################################################################################

##########################################################################################################################


# In[3]:


evaluation_history = np.load('Post_evaluation_history.npy')
beta_A  = np.load("Post_beta_scale_array.npy")
eta_A   = np.load("Post_eta_i_array.npy")
cur_A   = np.load("Post_cur_beta_array.npy")
x_A     = np.load("Post_x_array.npy")


# In[4]:


opt.update_design([mapping(x_A, eta_A, cur_A/beta_A)])  # cur_beta/beta_scale is the final beta in the optimization.

# In[ ]:


nfreq = 50
monitor_position = mp.Vector3(0, 0, 0.03)
monitor_size     = mp.Vector3(Sx, Sy, 0)


sim_empty = mp.Simulation(
    cell_size        = mp.Vector3(Sx, Sy ,Sz),
    boundary_layers  = pml_layers,
    geometry         = [],  
    sources          = source,
    default_material = Air,
    k_point          = kpoint,
    resolution       = resolution,
    extra_materials  = [Ag],
)


# Add flux monitors for RCP incident flux
incident_flux_monitor = sim_empty.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position, size=monitor_size))
sim_empty.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(z=-0.2), 1e-6))
trans_empty_fluxes = mp.get_fluxes(incident_flux_monitor)
freqs = mp.get_flux_freqs(incident_flux_monitor)
wavelengths = [1 / f for f in freqs]

np.save('TRAN/trans_empty_fluxes.npy',trans_empty_fluxes)
np.save('TRAN/trans_freqs.npy',freqs)
np.save('TRAN/trans_wavelengths.npy',wavelengths)
sim_empty.reset_meep()





# In[ ]:


opt.sim = mp.Simulation(
    cell_size        = mp.Vector3(Sx, Sy ,Sz),
    boundary_layers  = pml_layers,
    geometry         = geometry,  
    sources          = source,
    default_material = Air,
    k_point          = kpoint,
    resolution       = resolution,
    extra_materials  = [Ag],
)


# Add flux monitors for RCP incident flux
flux_monitor = opt.sim.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position, size=monitor_size))
opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(z=-0.2), 1e-6))
trans_fluxes = mp.get_fluxes(flux_monitor)
np.save('TRAN/trans_fluxes.npy',trans_fluxes)

opt.sim.reset_meep()






# In[ ]:




