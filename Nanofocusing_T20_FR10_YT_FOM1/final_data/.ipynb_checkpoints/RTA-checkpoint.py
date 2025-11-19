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
resolution = 100
design_region_resolution = int(resolution)

design_region_x_width  = 1   #100 nm
design_region_y_height = 1   #100 nm
design_region_z_height = 0.02  #20 nm or 10nm

pml_size = 1.0
pml_layers = [mp.PML(pml_size,direction=mp.Z)]

Sz_size = 0.6
Sx = design_region_x_width
Sy = design_region_y_height 
Sz = 2 * pml_size + design_region_z_height + Sz_size
cell_size = mp.Vector3(Sx, Sy, Sz)

wavelengths = np.array([1.55])     # wavelengths = np.array([1.5 ,1.55, 1.6])
frequencies = np.array([1 / 1.55])

nf = 1                 #3 #wavelengths Number

minimum_length = 0.01  # minimum length scale (microns)
eta_i = 0.5            # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55           # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e      # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)

Source_distance = -0.2

fcen   = 1 / 1.55
width  = 0.2  
fwidth = width * fcen
source_center = mp.Vector3(0,0,Source_distance)  
source_size   = mp.Vector3(design_region_x_width, design_region_y_height, 0)
src    = mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True)
source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)]   

Nx = int(design_region_resolution * design_region_x_width) + 1
Ny = int(design_region_resolution * design_region_y_height) + 1
Nz = 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny, Nz), Air, Ag, grid_type="U_MEAN")
design_region    = mpa.DesignRegion(
            design_variables,
            volume=mp.Volume(
            center=mp.Vector3(0,0,0),
            size=mp.Vector3(design_region_x_width, design_region_y_height, design_region_z_height),
            ),
)

def mapping(x, eta, beta):
    # filter
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_x_width,
        design_region_y_height,
        design_region_resolution,
    )
    # projection
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    projected_field = (npa.fliplr(projected_field) + projected_field) / 2
    projected_field = (npa.flipud(projected_field) + projected_field) / 2  # left-right symmetry    
    
    return projected_field.flatten()


geometry = [mp.Block(center=design_region.center, size=design_region.size, material=design_variables)]

kpoint = mp.Vector3()
sim = mp.Simulation(
    cell_size        = cell_size,
    boundary_layers  = pml_layers,
    geometry         = geometry,
    sources          = source,
    default_material = Air,
    k_point          = kpoint,
    symmetries       = [mp.Mirror(direction=mp.X)],
    resolution       = resolution,
    extra_materials  = [Ag],       # Introducing metal complex terms
)
#---------------------------------------------------------------------------------------------------#


#-----------------[4]---------------------------------#
monitor_position   = mp.Vector3(0, 0, 0)       # Focus position
monitor_size       = mp.Vector3(0.01,0.01,0.01)     # Focus Size//////0.11
FourierFields_x    = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ex,yee_grid=True)
FourierFields_y    = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ey,yee_grid=True)
FourierFields_z    = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ez,yee_grid=True)
ob_list            = [FourierFields_x,FourierFields_y,FourierFields_z]


def J(fields_x,fields_y,fields_z):
    ET_x = npa.abs(fields_x) ** 2
    ET_y = npa.abs(fields_y) ** 2
    ET_z = npa.abs(fields_z) ** 2
    ET = npa.sqrt( npa.mean(ET_x) + npa.mean(ET_y) + npa.mean(ET_z) )  
    return ET


opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=frequencies,
    decimation_factor = 1 ,           # KEY BUG!!
    maximum_run_time=50,
)
##########################################################################################################################


# In[3]:


evaluation_history = np.load('Post_evaluation_history.npy')
beta_A  = np.load("Post_beta_scale_array.npy")
eta_A   = np.load("Post_eta_i_array.npy")
cur_A   = np.load("Post_cur_beta_array.npy")
x_A     = np.load("Post_x_array.npy")


# In[4]:


opt.update_design([mapping(x_A, eta_A, cur_A)])  # cur_beta/beta_scale is the final beta in the optimization.

# In[ ]:


nfreq = 100
monitor_position   = mp.Vector3(0, 0,  0.1)
monitor_position_r = mp.Vector3(0, 0, -0.1)
monitor_size       = mp.Vector3(Sx, Sy, 0)


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
incident_flux_monitor       = sim_empty.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position   , size=monitor_size))
incident_reflection_monitor = sim_empty.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position_r , size=monitor_size))

sim_empty.run(until_after_sources =mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(z=0.1), 1e-3))

trans_empty_fluxes           = mp.get_fluxes(incident_flux_monitor)
incident_reflection_data     = sim_empty.get_flux_data(incident_reflection_monitor)
freqs = mp.get_flux_freqs(incident_flux_monitor)
wavelengths = [1 / f for f in freqs]

np.save('TRAN/trans_empty_fluxes.npy',trans_empty_fluxes)
np.save('TRAN/ref_empty_fluxes.npy'  ,incident_reflection_data)
np.save('TRAN/trans_freqs.npy'       ,freqs)
np.save('TRAN/trans_wavelengths.npy' ,wavelengths)
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
flux_monitor       = opt.sim.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position  , size=monitor_size))
reflection_monitor = opt.sim.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position_r, size=monitor_size))
opt.sim.load_minus_flux_data(reflection_monitor, incident_reflection_data)
opt.sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(z=0.1), 1e-3))
trans_fluxes      = mp.get_fluxes(flux_monitor)
reflection_fluxes = mp.get_fluxes(reflection_monitor)
np.save('TRAN/trans_fluxes.npy',trans_fluxes)
np.save('TRAN/reflection_fluxes.npy',reflection_fluxes)

T_FIN = np.array(trans_fluxes) / np.array(trans_empty_fluxes )
R_FIN = -np.array(reflection_fluxes) / np.array(trans_empty_fluxes)  # Negative due to flux direction convention
A_FIN = 1 - T_FIN  - R_FIN  # Absorption for RCP

np.save('TRAN/trans_final.npy',T_FIN)
np.save('TRAN/ref_final.npy'  ,R_FIN)
np.save('TRAN/abs_final.npy'  ,A_FIN)


opt.sim.reset_meep()






# In[ ]:




