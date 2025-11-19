import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from meep.materials import Ag
import os
import math

# In[2]:


dir_path = "3-D/analysis/LCPRCP0623_500nm_20nm_grad_500_rota"  
os.makedirs(f'{dir_path}/s_change',exist_ok=True) # Create Folder
os.makedirs(f'{dir_path}/final_data',exist_ok=True) # Create Folder
os.makedirs(f'{dir_path}/v_data',exist_ok=True)
mp.verbosity(0)
TiO2 = mp.Medium(index=2.6)
SiO2 = mp.Medium(index=1.44)
Si = mp.Medium(index=3.4)
Air = mp.Medium(index=1)
resolution = 100
design_region_resolution = int(resolution)

design_region_x_width  = 0.5   #100 nm
design_region_y_height = 0.5   #100 nm
design_region_z_height = 0.02  #20  nm

pml_size = 1.0
pml_layers = [mp.PML(pml_size,direction=mp.Z)]

Sz_size = 0.2
Sx = design_region_x_width
Sy = design_region_y_height 
Sz = 2 * pml_size + design_region_z_height + Sz_size
cell_size = mp.Vector3(Sx, Sy, Sz)

wavelengths = np.array([0.75])     # wavelengths = np.array([1.5 ,1.55, 1.6])
frequencies = np.array([1 / 0.75])

nf = 1                 #3 #wavelengths Number

minimum_length = 0.02  # minimum length scale (microns)
eta_i = 0.5            # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55           # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e      # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)

Source_distance = -0.05

fcen   = 1 / 0.75
width  = 0.2  
fwidth = width * fcen
df=fwidth
# 光源位置
src_z = -0.05
source_center = mp.Vector3(0, 0, src_z)
source_size   = mp.Vector3(design_region_x_width, design_region_y_height, 0)

# 設定脈衝的時間長度


lcp_sources = [
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),  
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=1.0j  # 左旋圓偏振 (LCP) +90° 相位
    )
]

rcp_sources = [
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=-1.0j  # 右旋圓偏振 (RCP) -90° 相位
    )
]


# 合併光源
sources = lcp_sources + rcp_sources


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
   # projected_field = (npa.flipud(projected_field) + projected_field) / 2  # left-right symmetry    
    
    return projected_field.flatten()


geometry = [mp.Block(center=design_region.center, size=design_region.size, material=design_variables)]

kpoint = mp.Vector3()
sim_LCP = mp.Simulation(
    cell_size        = cell_size,
    boundary_layers  = pml_layers,
    geometry         = geometry,
    sources          = lcp_sources,
    default_material = Air,
    k_point          = kpoint,
   # symmetries       = [mp.Mirror(direction=mp.X)],  # 與你原本 sim 一致
    resolution       = resolution,
    extra_materials  = [Ag],                          # 與你原本 sim 一致
)

# RCP 模擬
sim_RCP = mp.Simulation(
    cell_size        = cell_size,
    boundary_layers  = pml_layers,
    geometry         = geometry,
    sources          = rcp_sources,
    default_material = Air,
    k_point          = kpoint,
   # symmetries       = [mp.Mirror(direction=mp.X)],
    resolution       = resolution,
    extra_materials  = [Ag],
)

monitor_position = mp.Vector3(0, 0, 0.03)  # 監測區域中心
monitor_size = mp.Vector3(Sx, Sy, 0)       # 監測區域大小
sim_incident = mp.Simulation(
    cell_size        = cell_size,
    boundary_layers  = pml_layers,
    geometry=[],
    sources          = lcp_sources,
    default_material = Air,
    k_point          = kpoint,
   # symmetries       = [mp.Mirror(direction=mp.X)],  # 與你原本 sim 一致
    resolution       = resolution,
    extra_materials  = [Ag],                       
)
incident_flux_path = '3-D/img/LCPRCP0623_500nm_20nm_grad_500_rota/v_data/incident_flux_poynting.npy'
incident_flux = np.load(incident_flux_path)
print(f"[✓] Loaded incident flux: {incident_flux:.4f}")
weights_monitor_path = '3-D/img/LCPRCP0623_500nm_20nm_grad_500_rota/v_data/weights_monitor.npy'
weights_monitor = np.load(weights_monitor_path)
print(f"[✓] Loaded weights_monitor: {weights_monitor:}")

monitor_position = mp.Vector3(0, 0, 0.03)  # Above the structure (flux1)

monitor_size = mp.Vector3(Sx, Sy, 0)
# monitor_position_trans = mp.Vector3(0, 0, 0.03)
# monitor_size_trans = mp.Vector3(Sx, Sy, 0)
# monitor_position_refl = mp.Vector3(0, 0, -0.03)
# monitor_size_refl = mp.Vector3(Sx, Sy, 0)


# LCP Fourier fields
trans_Ex_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ex, yee_grid=True)
trans_Ey_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ey, yee_grid=True)
trans_Hx_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hx, yee_grid=True)
trans_Hy_LCP = mpa.FourierFields(sim_LCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hy, yee_grid=True)

# RCP Fourier fields
trans_Ex_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ex, yee_grid=True)
trans_Ey_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Ey, yee_grid=True)
trans_Hx_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hx, yee_grid=True)
trans_Hy_RCP = mpa.FourierFields(sim_RCP, mp.Volume(center=monitor_position, size=monitor_size), mp.Hy, yee_grid=True)

ob_list_LCP = [trans_Ex_LCP, trans_Ey_LCP, trans_Hx_LCP, trans_Hy_LCP]
ob_list_RCP = [trans_Ex_RCP, trans_Ey_RCP, trans_Hx_RCP, trans_Hy_RCP]

# Step 3: Define J Function
def J(Ex, Ey, Hx, Hy):
    # Extract field components for the single frequency
    Ex0 = Ex[0]
    Ey0 = Ey[0]
    Hx0 = Hx[0]
    Hy0 = Hy[0]
    
    # Trim arrays to the smallest common shape
    min_shape = tuple(npa.min([Ex0.shape, Ey0.shape, Hx0.shape, Hy0.shape], axis=0))
    Ex0 = Ex0[:min_shape[0], :min_shape[1]]
    Ey0 = Ey0[:min_shape[0], :min_shape[1]]
    Hx0 = Hx0[:min_shape[0], :min_shape[1]]
    Hy0 = Hy0[:min_shape[0], :min_shape[1]]
    
    # Compute Poynting flux density
    flux_density = 0.5 * npa.real(npa.conj(Ex0) * Hy0 - npa.conj(Ey0) * Hx0)
    
    # Load and trim weights
    if not os.path.exists(f'{dir_path}/v_data/weights_monitor.npy'):
        raise FileNotFoundError("Weights file not found. Ensure incident flux calculation is run first.")
    w = np.load(f'{dir_path}/v_data/weights_monitor.npy')
    w = w[:min_shape[0], :min_shape[1]]
    
    # Integrate flux
    total_flux = npa.sum(w * flux_density)
    
    # Normalize by incident flux
    if not os.path.exists(f'{dir_path}/v_data/incident_flux_poynting.npy'):
        raise FileNotFoundError("Incident flux file not found. Ensure incident flux calculation is run first.")
    incident_flux = np.load(f'{dir_path}/v_data/incident_flux_poynting.npy')
    transmittance = total_flux / (incident_flux + 1e-15)
    
    return transmittance

# Step 5: Set Up Optimization Problem
opt_LCP = mpa.OptimizationProblem(
    simulation=sim_LCP,
    objective_functions=[J],  # Use the created J_LCP
    objective_arguments=[trans_Ex_LCP, trans_Ey_LCP, trans_Hx_LCP, trans_Hy_LCP],
    design_regions=[design_region],
    frequencies=frequencies,
    decimation_factor=1,
    maximum_run_time=30,
)

# RCP Optimization: Minimize T_RCP
opt_RCP = mpa.OptimizationProblem(
    simulation=sim_RCP,
    objective_functions=[lambda Ex, Ey, Hx, Hy: -J(Ex, Ey, Hx, Hy)],  # Use the created J_RCP with negation
    objective_arguments=[trans_Ex_RCP, trans_Ey_RCP, trans_Hx_RCP, trans_Hy_RCP],
    design_regions=[design_region],
    frequencies=frequencies,
    decimation_factor=1,
    maximum_run_time=30,
)


def Plot_pre_fun(figsize, title, output_plane, path, Plot_save):
    plt.figure(figsize=figsize)
    plt.title(title)
    opt_RCP.plot2D(True, output_plane=output_plane)
     # **這裡加入 xlim 和 ylim 設定**
    if Plot_save:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Detector_x_y', 
             output_plane = mp.Volume(center=monitor_position, size=mp.Vector3(Sx,Sy,0)), 
             path         = f'{dir_path}/Pre_Detector_x_y.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Source', 
             output_plane = mp.Volume(center=source_center, size=mp.Vector3(Sx,Sy,0)), 
             path         = f'{dir_path}/Pre_Source_x_y.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Structure', 
             output_plane = mp.Volume(center=design_region.center, size=mp.Vector3(Sx,Sy,0)), 
             path         = f'{dir_path}/Pre_Structure_x_y.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Sx_Sz', 
             output_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(Sx,0,Sz)), 
             path         = f'{dir_path}/Pre_Sx_Sz.png', 
             Plot_save    = 1 )

Plot_pre_fun(figsize      = (5, 10), 
             title        = 'Sy_Sz', 
             output_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(0,Sy,Sz)), 
             path         = f'{dir_path}/Pre_Sy_Sz_d.png', 
             Plot_save    = 1 )

#---------------------------------------------------------------------------------------------------#




evaluation_history = []
cur_iter = [0]

def f(v, gradient, beta):
    print(f"Current iteration: {cur_iter[0] + 1}")
    
    # Map design variables to structure
    x_avg = mapping(v, eta_i, beta)
    
    # Simulate LCP and RCP
    f0_L, dJ_du_L = opt_LCP([x_avg])  # T_LCP
    f0_R, dJ_du_R = opt_RCP([x_avg])  # -T_RCP
    
    # Compute FOM (T_LCP - T_RCP)
    T_LCP = f0_L
    T_RCP = -f0_R  # Convert back to positive transmittance
    f0 = T_LCP - T_RCP  # Circular dichroism
    
    # Apply PCGrad to resolve gradient conflicts
    dJ_du_L = np.array(dJ_du_L)
    dJ_du_R = np.array(dJ_du_R)
    
    # Combine adjusted gradients for the FOM
    dJ_du = (dJ_du_L + dJ_du_R) *500  # Average the adjusted gradients
    
    print(f"LCP Transmittance: {T_LCP:.6f}")
    print(f"RCP Transmittance: {T_RCP:.6f}")
    print(f"FOM (CD): {f0:.6f}")
    
    
    # Backpropagate the adjusted gradient
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(v, eta_i, beta, dJ_du)
    
    evaluation_history.append(np.real(f0))
    
    # Plot current structure
    if mp.am_master():
        plt.figure(figsize=(5, 5))
        opt_LCP.plot2D(
            plot_sources_flag=False,
            plot_monitors_flag=False,
            plot_boundaries_flag=False,
            output_plane=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(design_region_x_width, design_region_y_height, 0)
            )
        )
        plt.title(f"Structure (Iteration {cur_iter[0] + 1})")
        plt.savefig(f'{dir_path}/s_change/s_change{cur_iter[0]:03d}.png')
        plt.close()
    
    # Save data
    np.save(f'{dir_path}/v_data/Post_v_array{cur_iter[0]:03d}.npy', v)
    np.save(f'{dir_path}/v_data/Post_x_array{cur_iter[0]:03d}.npy', x_avg)
    np.save(f'{dir_path}/v_data/Post_eta_i_array{cur_iter[0]:03d}.npy', eta_i)
    np.save(f'{dir_path}/v_data/Post_cur_beta_array{cur_iter[0]:03d}.npy', cur_beta)
    np.save(f'{dir_path}/v_data/Post_beta_scale_array{cur_iter[0]:03d}.npy', beta_scale)
    np.save(f'{dir_path}/v_data/T_LCP_array{cur_iter[0]:03d}.npy', T_LCP)
    np.save(f'{dir_path}/v_data/T_RCP_array{cur_iter[0]:03d}.npy', T_RCP)
    np.save(f'{dir_path}/v_data/Post_f0_L_array{cur_iter[0]:03d}.npy', f0_L)
    np.save(f'{dir_path}/v_data/Post_f0_R_array{cur_iter[0]:03d}.npy', f0_R)
    np.save(f'{dir_path}/v_data/Post_FOM_array{cur_iter[0]:03d}.npy', f0)
    np.save(f'{dir_path}/v_data/Post_grad_total_array{cur_iter[0]:03d}.npy', dJ_du)
    np.save(f'{dir_path}/v_data/Post_grad_LCP_array{cur_iter[0]:03d}.npy', dJ_du_L)
    np.save(f'{dir_path}/v_data/Post_grad_RCP_array{cur_iter[0]:03d}.npy', dJ_du_R)
    cur_iter[0] += 1
    return np.real(f0)

import numpy as np
import matplotlib.pyplot as plt
import math
import nlopt

algorithm = nlopt.LD_MMA
n = Nx * Ny * Nz
np.random.seed(0) 
x = np.clip(np.random.normal(loc=0.5, scale=0.1, size=n), 0, 1)
lb = np.zeros((n,))
ub = np.ones((n,))


beta_A = np.load("3-D/img/LCPRCP0623_500nm_20nm_grad_500_rota/final_data/Post_beta_scale_array.npy")
eta_A = np.load("3-D/img/LCPRCP0623_500nm_20nm_grad_500_rota/final_data/Post_eta_i_array.npy")
cur_A = np.load("3-D/img/LCPRCP0623_500nm_20nm_grad_500_rota/final_data/Post_cur_beta_array.npy")
x_A = np.load("3-D/img/LCPRCP0623_500nm_20nm_grad_500_rota/final_data/Post_x_array.npy")
print(beta_A)
print(eta_A)
print(cur_A)
print(x_A)


opt_LCP.update_design([mapping(x_A, eta_A, cur_A / beta_A)])
plt.figure()
ax = plt.gca()
opt_LCP.plot2D(
    False,
    ax=ax,
    plot_sources_flag=False,
    plot_monitors_flag=False,
    plot_boundaries_flag=False,
    output_plane=mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(design_region_x_width, design_region_y_height, 0))
)
circ = Circle((2, 2), minimum_length / 2)
ax.add_patch(circ)
ax.axis("off")
plt.savefig(f'{dir_path}/Post_Structure_x_y.png')
plt.close()

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# Step 1: Define broadband source for visible spectrum
lambda_min = 0.360  # 360 nm
lambda_max = 0.830  # 830 nm
fcen = (1 / lambda_min + 1 / lambda_max) / 2  # Center frequency
fwidth = abs(1 / lambda_min - 1 / lambda_max)  # Frequency width
nfreq = 500  # Number of frequency points for flux monitors
src_z = -0.05
source_center = mp.Vector3(0, 0, src_z)
source_size = mp.Vector3(design_region_x_width, design_region_y_height, 0)

# Define Gaussian source
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True)

# LCP sources
lcp_sources = [
    mp.Source(
        src=src,
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=src,
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=1.0j  # +90° phase for LCP
    )
]

# RCP sources
rcp_sources = [
    mp.Source(
        src=src,
        component=mp.Ex,
        center=source_center,
        size=source_size,
        amplitude=1.0
    ),
    mp.Source(
        src=src,
        component=mp.Ey,
        center=source_center,
        size=source_size,
        amplitude=-1.0j  # -90° phase for RCP
    )
]

# Step 2: Calculate incident flux for LCP
sim_incident_lcp = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=[],  # No geometry for incident flux
    sources=lcp_sources,
    default_material=Air,
    k_point=mp.Vector3(0, 0, 0),
    resolution=resolution,
    extra_materials=[Ag],
)

# Add flux monitors for LCP incident flux
incident_monitor_position = mp.Vector3(0, 0, 0.03)
incident_monitor_size = mp.Vector3(Sx, Sy, 0)
lcp_incident_flux_monitor = sim_incident_lcp.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=incident_monitor_position, size=incident_monitor_size))
lcp_incident_reflection_monitor = sim_incident_lcp.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, -0.06), size=incident_monitor_size))

# Run LCP incident flux simulation
sim_incident_lcp.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(z=-0.05), 1e-6))
lcp_incident_fluxes = mp.get_fluxes(lcp_incident_flux_monitor)
lcp_incident_reflection_data = sim_incident_lcp.get_flux_data(lcp_incident_reflection_monitor)  # Save incident reflection data
freqs = mp.get_flux_freqs(lcp_incident_flux_monitor)
wavelengths = [1 / f for f in freqs]

# Save LCP incident flux
np.save(f'{dir_path}/v_data/lcp_incident_flux_broadband.npy', lcp_incident_fluxes)

# Clean up LCP incident simulation
sim_incident_lcp.reset_meep()

# Step 3: Calculate incident flux for RCP
sim_incident_rcp = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=[],  # No geometry for incident flux
    sources=rcp_sources,
    default_material=Air,
    k_point=mp.Vector3(0, 0, 0),
    resolution=resolution,
    extra_materials=[Ag],
)

# Add flux monitors for RCP incident flux
rcp_incident_flux_monitor = sim_incident_rcp.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=incident_monitor_position, size=incident_monitor_size))
rcp_incident_reflection_monitor = sim_incident_rcp.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, -0.06), size=incident_monitor_size))

# Run RCP incident flux simulation
sim_incident_rcp.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(z=-0.05), 1e-6))
rcp_incident_fluxes = mp.get_fluxes(rcp_incident_flux_monitor)
rcp_incident_reflection_data = sim_incident_rcp.get_flux_data(rcp_incident_reflection_monitor)  # Save incident reflection data

# Save RCP incident flux
np.save(f'{dir_path}/v_data/rcp_incident_flux_broadband.npy', rcp_incident_fluxes)

# Clean up RCP incident simulation
sim_incident_rcp.reset_meep()

# Step 4: LCP simulation with structure
opt_LCP.sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=lcp_sources,
    default_material=Air,
    resolution=resolution,
    k_point=mp.Vector3(0, 0, 0),
    extra_materials=[Ag],
)

# Add flux monitors for LCP
monitor_position = mp.Vector3(0, 0, 0.03)
monitor_size = mp.Vector3(Sx, Sy, 0)
lcp_flux_monitor = opt_LCP.sim.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position, size=monitor_size))
lcp_reflection_monitor = opt_LCP.sim.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, -0.06), size=monitor_size))

# Load incident reflection data and run LCP simulation
opt_LCP.sim.load_minus_flux_data(lcp_reflection_monitor, lcp_incident_reflection_data)
opt_LCP.sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(z=-0.05), 1e-6))
lcp_fluxes = mp.get_fluxes(lcp_flux_monitor)
lcp_reflection_fluxes = mp.get_fluxes(lcp_reflection_monitor)  # This is now the scattered flux

# Calculate LCP transmission, reflection, and absorption
T_LCP = np.array(lcp_fluxes) / np.array(lcp_incident_fluxes)
R_LCP = -np.array(lcp_reflection_fluxes) / np.array(lcp_incident_fluxes)  # Negative due to flux direction convention
A_LCP = 1 - T_LCP - R_LCP  # Absorption for LCP

# Step 5: RCP simulation with structure
opt_LCP.sim.reset_meep()
opt_RCP.sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=rcp_sources,
    default_material=Air,
    resolution=resolution,
    k_point=mp.Vector3(0, 0, 0),
    extra_materials=[Ag],
)

# Add flux monitors for RCP
rcp_flux_monitor = opt_RCP.sim.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=monitor_position, size=monitor_size))
rcp_reflection_monitor = opt_RCP.sim.add_flux(fcen, fwidth, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, -0.06), size=monitor_size))

# Load incident reflection data and run RCP simulation
opt_RCP.sim.load_minus_flux_data(rcp_reflection_monitor, rcp_incident_reflection_data)
opt_RCP.sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(z=-0.05), 1e-6))
rcp_fluxes = mp.get_fluxes(rcp_flux_monitor)
rcp_reflection_fluxes = mp.get_fluxes(rcp_reflection_monitor)  # This is now the scattered flux

# Calculate RCP transmission, reflection, and absorption
T_RCP = np.array(rcp_fluxes) / np.array(rcp_incident_fluxes)
R_RCP = -np.array(rcp_reflection_fluxes) / np.array(rcp_incident_fluxes)  # Negative due to flux direction convention
A_RCP = 1 - T_RCP - R_RCP  # Absorption for RCP

# Calculate transmission, reflection, and absorption differences
transmission_diff = T_LCP - T_RCP
reflection_diff = R_LCP - R_RCP
absorption_diff = A_LCP - A_RCP

# Save results
np.save(f'{dir_path}/final_data/Post_T_LCP_broadband.npy', T_LCP)
np.save(f'{dir_path}/final_data/Post_T_RCP_broadband.npy', T_RCP)
np.save(f'{dir_path}/final_data/Post_R_LCP_broadband.npy', R_LCP)
np.save(f'{dir_path}/final_data/Post_R_RCP_broadband.npy', R_RCP)
np.save(f'{dir_path}/final_data/Post_A_LCP_broadband.npy', A_LCP)
np.save(f'{dir_path}/final_data/Post_A_RCP_broadband.npy', A_RCP)
np.save(f'{dir_path}/final_data/Post_transmission_diff_broadband.npy', transmission_diff)
np.save(f'{dir_path}/final_data/Post_reflection_diff_broadband.npy', reflection_diff)
np.save(f'{dir_path}/final_data/Post_absorption_diff_broadband.npy', absorption_diff)
np.save(f'{dir_path}/final_data/wavelengths.npy', wavelengths)

# Transmission plot
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, T_LCP, label='T_LCP', color='blue')
plt.plot(wavelengths, T_RCP, label='T_RCP', color='orange')
plt.plot(wavelengths, transmission_diff, label='T_LCP - T_RCP', color='green')
plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Transmission', fontsize=12)
plt.title('Transmission Spectra: LCP vs RCP', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.8)
plt.savefig(f'{dir_path}/transmission_spectro.png', dpi=300, bbox_inches='tight')
plt.close()

# Reflection plot
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, R_LCP, label='R_LCP', color='blue')
plt.plot(wavelengths, R_RCP, label='R_RCP', color='orange')
plt.plot(wavelengths, reflection_diff, label='R_LCP - R_RCP', color='green')
plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Reflection', fontsize=12)
plt.title('Reflection Spectra: LCP vs RCP', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.8)
plt.savefig(f'{dir_path}/reflection_spectro.png', dpi=300, bbox_inches='tight')
plt.close()

# Absorption plot
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, A_LCP, label='A_LCP', color='blue')
plt.plot(wavelengths, A_RCP, label='A_RCP', color='orange')
plt.plot(wavelengths, absorption_diff, label='A_LCP - A_RCP', color='green')
plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Absorption', fontsize=12)
plt.title('Absorption Spectra: LCP vs RCP', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.8)
plt.savefig(f'{dir_path}/absorption_spectro.png', dpi=300, bbox_inches='tight')
plt.close()

# Mixed T, R, A plot
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))

# Transmission
plt.plot(wavelengths, T_LCP, label='T_LCP', color='blue', linestyle='-', marker='o', markevery=20)
plt.plot(wavelengths, T_RCP, label='T_RCP', color='blue', linestyle='--', marker='x', markevery=20)

# Reflection
plt.plot(wavelengths, R_LCP, label='R_LCP', color='red', linestyle='-', marker='s', markevery=20)
plt.plot(wavelengths, R_RCP, label='R_RCP', color='red', linestyle='--', marker='d', markevery=20)

# Absorption
plt.plot(wavelengths, A_LCP, label='A_LCP', color='green', linestyle='-', marker='^', markevery=20)
plt.plot(wavelengths, A_RCP, label='A_RCP', color='green', linestyle='--', marker='v', markevery=20)

plt.xlabel('Wavelength (µm)', fontsize=12)
plt.ylabel('Fraction', fontsize=12)
plt.title('Transmission, Reflection, and Absorption Spectra: LCP vs RCP', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.8)
plt.savefig(f'{dir_path}/mixed_spectro.png', dpi=300, bbox_inches='tight')
plt.close()