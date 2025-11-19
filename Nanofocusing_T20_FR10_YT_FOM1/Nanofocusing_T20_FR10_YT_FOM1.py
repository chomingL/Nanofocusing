#-----------------[1]---------------------------------#
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
import shutil
#---------------------------------------------------------------------------------------------------#

#-----------------[2]---------------------------------#
filename = os.path.basename(__file__)
filename_without = os.path.splitext(filename)[0]  
os.makedirs(f's_change'  ,exist_ok=True) # Create Folder
os.makedirs(f'final_data',exist_ok=True) # Create Folder
os.makedirs(f'v_data'    ,exist_ok=True)
shutil.copy("img/Test_analysis_file/DFT_empty.ipynb"    ,f'{dir_path}/final_data')
shutil.copy("img/Test_analysis_file/DFT_post.ipynb"     ,f'{dir_path}/final_data')
shutil.copy("img/Test_analysis_file/DFT.ipynb"          ,f'{dir_path}/final_data')
shutil.copy("img/Test_analysis_file/E_Q.ipynb"          ,f'{dir_path}/final_data')
shutil.copy("img/Test_analysis_file/ST.ipynb"           ,f'{dir_path}/final_data')
shutil.copy("img/Test_analysis_file/Animate_Structure.ipynb",f'{dir_path}/v_data')
mp.verbosity(0)
TiO2 = mp.Medium(index=2.6)
SiO2 = mp.Medium(index=1.44)
Si = mp.Medium(index=3.4)
Air = mp.Medium(index=1)
Plot_save = 1
#---------------------------------------------------------------------------------------------------#


#-----------------[3]---------------------------------#
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
#---------------------------------------------------------------------------------------------------#

#-----------------[5]---------------------------------#

def Plot_pre_fun(figsize, title, output_plane, path, Plot_save):
    plt.figure(figsize=figsize)
    plt.title(title)
    opt.plot2D(True,output_plane = output_plane)
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




#-----------------[6]---------------------------------#


evaluation_history = []
cur_iter = [0]


def f(v, gradient, beta):
    print("Current iteration: {}".format(cur_iter[0] + 1))

    f0, dJ_du = opt([mapping(v, eta_i, beta)])  # compute objective and gradient

    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, beta, dJ_du,
        )  # backprop
    evaluation_history.append(np.real(f0))
    #print(gradient)
    if mp.am_master():
        plt.figure(figsize=(5, 10))
        ax = plt.gca()
        opt.plot2D(
            False,
            ax=ax,
            plot_sources_flag    = False,
            plot_monitors_flag   = False,
            plot_boundaries_flag = False,
            output_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(design_region_x_width, design_region_y_height, 0))
        )
        circ = Circle((2, 2), minimum_length / 2)
        ax.add_patch(circ)
        ax.axis("off")
        
        plt.savefig(f'{dir_path}/s_change/s_change{cur_iter[0] :03d}.png')
        plt.close()

    cur_iter[0] = cur_iter[0] + 1
    print(f0)
    np.save(f'{dir_path}/v_data/Post_v_array{cur_iter[0] :03d}.npy', v)
    
    np.save(f'{dir_path}/v_data/Post_x_array{cur_iter[0] :03d}.npy', x)
    np.save(f'{dir_path}/v_data/Post_eta_i_array{cur_iter[0] :03d}.npy', eta_i)
    np.save(f'{dir_path}/v_data/Post_cur_beta_array{cur_iter[0] :03d}.npy', cur_beta)
    np.save(f'{dir_path}/v_data/Post_beta_scale_array{cur_iter[0] :03d}.npy', beta_scale)

    
    return np.real(f0)
#---------------------------------------------------------------------------------------------------#

#-----------------[7]---------------------------------#


algorithm = nlopt.LD_MMA
#algorithm = nlopt.LN_COBYLA
n = Nx * Ny * Nz # number of parameters

# Initial guess
a = np.ones((218,)) * 0.5
b = np.ones((5,)) * 0.5
c = np.ones((218,)) * 0.5
x1 = np.append(np.append(a,b),c)


x = np.ones((n,)) * 0.5

#x = np.random.uniform(0,1,n)

# lower and upper bounds
lb = np.zeros((Nx * Ny * Nz,))
ub = np.ones((Nx * Ny * Nz,))

cur_beta      = 4    # 4
beta_scale    = 2    # 2
num_betas     = 12   # 12
update_factor = 15   # 12
ftol = 1e-5
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor)
    solver.set_ftol_rel(ftol)
    x[:] = solver.optimize(x)
    cur_beta = cur_beta * beta_scale

#---------------------------------------------------------------------------------------------------#

#-----------------[8]---------------------------------#
np.save(f'{dir_path}/final_data/Post_evaluation_history.npy', evaluation_history)
np.save(f'{dir_path}/final_data/Post_x_array.npy', x)
np.save(f'{dir_path}/final_data/Post_eta_i_array.npy', eta_i)
np.save(f'{dir_path}/final_data/Post_cur_beta_array.npy', cur_beta)
np.save(f'{dir_path}/final_data/Post_beta_scale_array.npy', beta_scale)


plt.figure()
plt.plot(evaluation_history, "o-")

for i in range(1,13):
    plt.axvline(x=12*i, color='purple', linestyle='--', label='x=3')
plt.grid(True)
plt.xlim(0)
plt.ylim(0)
plt.xlabel("Iteration")
plt.ylabel("FOM")
if Plot_save:
    plt.savefig(f'{dir_path}/Post_Fom_change.png')
    plt.close()
else:
    plt.show()
#---------------------------------------------------------------------------------------------------#


#-----------------[9]---------------------------------#

opt.update_design([mapping(x, eta_i, cur_beta/beta_scale)])    # cur_beta/beta_scale is the final beta in the optimization.
plt.figure()
ax = plt.gca()
opt.plot2D(
    False,
    ax=ax,
    plot_sources_flag=False,
    plot_monitors_flag=False,
    plot_boundaries_flag=False,
    output_plane = mp.Volume(center=mp.Vector3(0,0,0.01), size=mp.Vector3(design_region_x_width, design_region_y_height, 0))
)
circ = Circle((2, 2), minimum_length / 2)
ax.add_patch(circ)
ax.axis("off")
plt.savefig(f'{dir_path}/Post_Structure_x_y.png')
plt.close()
#---------------------------------------------------------------------------------------------------#


print('over1')
















