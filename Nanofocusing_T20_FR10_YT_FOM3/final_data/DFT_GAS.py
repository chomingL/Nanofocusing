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


# In this example, we consider wave propagation through a simple 1d nonlinear medium with a non-zero Kerr susceptibility 

# In[2]:


os.makedirs('DFT_GAS',exist_ok=True) # Create Folder
os.makedirs('DFT_mode_volume',exist_ok=True) # Create Folder




# In[3]:

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

minimum_length = 0.02  # minimum length scale (microns)
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

monitor_position   = mp.Vector3(0, 0, 0)       # Focus position
monitor_size       = mp.Vector3(0.01,0.01,design_region_z_height)     # Focus Size//////0.11
FourierFields_x    = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ex,yee_grid=True)
FourierFields_y    = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ey,yee_grid=True)
FourierFields_z    = mpa.FourierFields(sim,mp.Volume(center=monitor_position,size=monitor_size),mp.Ez,yee_grid=True)
ob_list            = [FourierFields_x,FourierFields_y,FourierFields_z]


def J(fields_x,fields_y,fields_z):
    ET_x = 0
    ET_y = 0
    ET_z = 0
    ET_x_length = fields_x.shape[1]
    ET_y_length = fields_y.shape[1]
    ET_z_length = fields_z.shape[1]
    for k in range(0,ET_x_length):
        ET_x = ET_x + npa.abs(fields_x[:,k]) ** 2
    for k in range(0,ET_y_length):
        ET_y = ET_y + npa.abs(fields_y[:,k]) ** 2
    for k in range(0,ET_z_length):
        ET_z = ET_z + npa.abs(fields_z[:,k]) ** 2
    ET = ( (npa.mean(ET_x)) + (npa.mean(ET_y)) + (npa.mean(ET_z)) ) ** (1/2)  
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


# In[4]:


evaluation_history = np.load('Post_evaluation_history.npy')
beta_A  = np.load("Post_beta_scale_array.npy")
eta_A   = np.load("Post_eta_i_array.npy")
cur_A   = np.load("Post_cur_beta_array.npy")
x_A     = np.load("Post_x_array.npy")


# In[5]:



# In[6]:

opt.update_design([mapping(x_A, eta_A, cur_A/beta_A)])  # cur_beta/beta_scale is the final beta in the optimization.

# In[7]:


opt.sim = mp.Simulation(
    cell_size=mp.Vector3(Sx, Sy ,Sz),
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=source,
    default_material=Air,
    resolution=resolution,
    k_point=kpoint,
    #symmetries=[mp.Mirror(direction=mp.X)],
    extra_materials = [Ag],
)
#src = mp.ContinuousSource(frequency=frequencies[0], fwidth=0.01 ,is_integrated=True)
src    = mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True)
source = [mp.Source(src, component=mp.Ey, size=source_size, center=source_center)] 
opt.sim.change_sources(source)

dft_fields_xz = opt.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],
                                fcen,0,1,
                                center=mp.Vector3(),
                                size=mp.Vector3(Sx,0,Sz-2 * pml_size),
                                yee_grid=False
                                )

dft_fields_yz = opt.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],
                                fcen,0,1,
                                center=mp.Vector3(),
                                size=mp.Vector3(0,Sy,Sz-2 * pml_size),
                                yee_grid=False
                                )

dft_fields_xyz12 = opt.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],
                                fcen,0,1,
                                center=mp.Vector3(0,0,0.01),
                                size=mp.Vector3(Sx,Sy,0),
                                yee_grid=False
                                )

dft_fields_xyz00 = opt.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],
                                fcen,0,1,
                                center=mp.Vector3(0,0,0),
                                size=mp.Vector3(Sx,Sy,0),
                                yee_grid=False 
                                )

dft_fields_x0yz0 = opt.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],
                                fcen,0,1,
                                center=mp.Vector3(0,0,0.01),
                                size=mp.Vector3(0,Sy,0),
                                yee_grid=False
                                )

dft_fields_mode_volume = opt.sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],
                                fcen,0,1,
                                center=mp.Vector3(0,0,0),
                                size=mp.Vector3(Sx,Sy,0.4),
                                yee_grid=False 
                                )


#opt.sim.run(until_after_sources = 0)
opt.sim.run(until_after_sources=mp.stop_when_dft_decayed(
            tol=1e-11,
            minimum_run_time=0,
            maximum_run_time=50))


# In[8]:


Ex_xz = opt.sim.get_dft_array(dft_fields_xz,mp.Ex,0)
Ey_xz = opt.sim.get_dft_array(dft_fields_xz,mp.Ey,0)
Ez_xz = opt.sim.get_dft_array(dft_fields_xz,mp.Ez,0)
np.save('DFT_GAS/Dft_GAS_Ex_xz.npy',Ex_xz)
np.save('DFT_GAS/Dft_GAS_Ey_xz.npy',Ey_xz)
np.save('DFT_GAS/Dft_GAS_Ez_xz.npy',Ez_xz)

Ex_yz = opt.sim.get_dft_array(dft_fields_yz,mp.Ex,0)
Ey_yz = opt.sim.get_dft_array(dft_fields_yz,mp.Ey,0)
Ez_yz = opt.sim.get_dft_array(dft_fields_yz,mp.Ez,0)
np.save('DFT_GAS/Dft_GAS_Ex_yz.npy',Ex_yz)
np.save('DFT_GAS/Dft_GAS_Ey_yz.npy',Ey_yz)
np.save('DFT_GAS/Dft_GAS_Ez_yz.npy',Ez_yz)

Ex01 = opt.sim.get_dft_array(dft_fields_xyz12,mp.Ex,0)
Ey01 = opt.sim.get_dft_array(dft_fields_xyz12,mp.Ey,0)
Ez01 = opt.sim.get_dft_array(dft_fields_xyz12,mp.Ez,0)
np.save('DFT_GAS/Dft_GAS_Ex_xyz1.npy',Ex01)
np.save('DFT_GAS/Dft_GAS_Ey_xyz1.npy',Ey01)
np.save('DFT_GAS/Dft_GAS_Ez_xyz1.npy',Ez01)

Ex00 = opt.sim.get_dft_array(dft_fields_xyz00,mp.Ex,0)
Ey00 = opt.sim.get_dft_array(dft_fields_xyz00,mp.Ey,0)
Ez00 = opt.sim.get_dft_array(dft_fields_xyz00,mp.Ez,0)
np.save('DFT_GAS/Dft_GAS_Ex_xyz0.npy',Ex00)
np.save('DFT_GAS/Dft_GAS_Ey_xyz0.npy',Ey00)
np.save('DFT_GAS/Dft_GAS_Ez_xyz0.npy',Ez00)

Ex_mode_volume = opt.sim.get_dft_array(dft_fields_mode_volume,mp.Ex,0)
Ey_mode_volume = opt.sim.get_dft_array(dft_fields_mode_volume,mp.Ey,0)
Ez_mode_volume = opt.sim.get_dft_array(dft_fields_mode_volume,mp.Ez,0)
np.save('DFT_mode_volume/DFT_mode_volume_Ex_xyz0.npy',Ex_mode_volume)
np.save('DFT_mode_volume/DFT_mode_volume_Ey_xyz0.npy',Ey_mode_volume)
np.save('DFT_mode_volume/DFT_mode_volume_Ez_xyz0.npy',Ez_mode_volume)


# In[ ]:




