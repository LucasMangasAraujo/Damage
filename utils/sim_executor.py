import numpy as np
import os
import utils.pre_processing as pre
import utils.post_processing as post
from .network_class import NetworkClass, FillerNetworkClass
from .loading import deformation_gradient
from pathlib import Path

def runsim_filler(geometry_file, model, params, dim, nFillers, filler_radius,
                    filler_epsilon, stiffness_ratio, angle_model, angle_params,
                    loading, stretch_array, stretch_increment, data_file):
    """
    Run full simulation.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        nFillers (int):
        filler_radius(float):
        filler_epsilon(float):
        stiffness_ratio (float):
        angle_model (str):
        angle_params(tuple):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        data_file (str):
        
    Outputs:
        stress_array (ndarray): Array with all the stress results
    """
    
    # Unpack parameters tuple
    bKuhn, NKuhn, nub3 = params
    
    # Relax as generated network
    relax_as_generated_DN(geometry_file, model, params, dim)
    DN_gen = NetworkClass("temp.dat", "test.res", "main.in")
    computational_params = DN_gen.get_computational_params(params) ## extract computational params
    
    # Place fillers in the networ and assign bond types
    Nodes, Bonds, bond_flags, Angles, angle_to_pair, Boundary, selected_nodes = pre.create_fillers(nFillers, filler_radius, filler_epsilon)
    BondTypes, rest_lengths = pre.assign_bond_types(bond_flags, filler_radius, filler_epsilon, NKuhn, stiffness_ratio)
    
    # Write data file
    pre.write_data_file(data_file, Nodes, Bonds, Angles, Boundary, BondTypes, model, 
                            computational_params, rest_lengths ,angle_model, angle_params)
    
    # Initialise output array
    stress_array = [] ## for now a list
    
    # Run relaxation step ...
    print("Starting simulation for network in file %s..." %geometry_file)
    print(100 * "=")
    bond_coeffs_lines = []
    
    if model != '1':
        ## When hybrid bond style is used, we need to store the bond coefficients
        ## lines.
        bond_coeff_lines = NetworkClass.get_bond_coeffs(data_file)
    else:
        ## Otherwise proceed stating an empty list
        bond_coeff_lines = []
    
    run_relaxation_hybrid(dim, data_file, Boundary, model, angle_model, bond_coeff_lines)
    DN = FillerNetworkClass(data_file, "test.res","main_hybrid.in") ## Netwotk object
    cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
    stress_array.append(cauchy_stress)
    radii_deviations = DN.get_filler_radii_deviations(bond_flags, selected_nodes, filler_radius)
    angles_deviations, max_dev, min_dev = DN.get_filler_angles_deviations(angle_to_pair)
    
    # Check for overlaps and spheres that might have left the domain
    is_overlaped_array = DN.any_filler_overlap(selected_nodes, filler_radius, filler_epsilon)
    if np.any(is_overlaped_array):
        print("Filler overlap occured!!")
        breakpoint()
    else:
        print("No filler overlapping detected in relaxation step.")
    initial_box = DN.get_box_lengths()
    is_missing_array = DN.any_filler_missing(selected_nodes, initial_box, filler_epsilon)
    
    if np.any(is_missing_array):
        print("There are missing fillers !!!!")
        breakpoint()
    else:
        print("Missing fillers were not detected")
    
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("max radii deviation is %g, while the min is %g" %(max(radii_deviations), min(radii_deviations)))
    print("max avg deviation (in degrees) is %g, while the min one is %g" %(max_dev, min_dev) )
    print(100 * "=")
    
    # Apply deformation history
    for i in range(1, len(stretch_array)):
        print(100 * "=")
        
        ## Run deformatio step
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main_hybrid.in')
        
        ## Check for simulation errors.
        if err:
            breakpoint()
        
        ## Reconstruct data if needed
        if model != '1':
            post.rewrite_data_file(bond_coeff_lines, data_file)
        
        ## Calculate DN information
        DN = FillerNetworkClass(data_file, "test.res","main_hybrid.in") ## Netwotk object
        F = deformation_gradient(loading, stretch_array[i])
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        radii_deviations = DN.get_filler_radii_deviations(bond_flags, selected_nodes, filler_radius)
        angles_deviations, max_dev, min_dev = DN.get_filler_angles_deviations(angle_to_pair)
        
        
        ## Check for potential overlaps and missing fillers
        is_overlaped_array = DN.any_filler_overlap(selected_nodes, filler_radius, filler_epsilon)
        if np.any(is_overlaped_array):
            print("Filler overlap occured!!")
            breakpoint()
        else:
            print("No filler overlapping detected in current deformation step.")
        
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("max radii deviation is %g, while the min is %g" %(max(radii_deviations), min(radii_deviations)))
        print("max avg deviation (in degrees) is %g, while the min one is %g" %(max_dev, min_dev) )
        
        ## Append current stress to the stress array
        stress_array.append(cauchy_stress)
        print(100 * "=")
        
    print("Finished simulation for network in file %s!" %geometry_file)
    print("\n\n")
    
    # Convert stress array to ndarray
    stress_array = NetworkClass.render_stress_units(np.array(stress_array), bKuhn)
    
    return stress_array



def runsim_filler_rep(geometry_file, model, params, dim, nFillers, filler_radius,
                filler_epsilon, stiffness_ratio, angle_model, angle_params,
                loading, stretch_array, stretch_increment, data_file, 
                folder_names):
    """
    Run simulation for representative network.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        nFillers (int):
        filler_radius(float):
        filler_epsilon(float):
        stiffness_ratio (float):
        angle_model (str):
        angle_params(tuple):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        data_file (str):
        folder_names (tuple): sequence of strings containing the path to place the 
                             data file.
        
    Outputs:
        stress_array (ndarray): Array with all the stress results
    """
    
    # Unpack parameters tuple
    bKuhn, NKuhn, nub3 = params
    
    # Relax as generated network
    relax_as_generated_DN(geometry_file, model, params, dim)
    DN_gen = NetworkClass("temp.dat", "test.res", "main.in")
    computational_params = DN_gen.get_computational_params(params) ## extract computational params
    
    # Place fillers in the networ and assign bond types
    Nodes, Bonds, bond_flags, Angles, angle_to_pair, Boundary, selected_nodes = pre.create_fillers(nFillers, filler_radius, filler_epsilon)
    BondTypes, rest_lengths = pre.assign_bond_types(bond_flags, filler_radius, filler_epsilon, NKuhn, stiffness_ratio)
    
    # Write data file
    pre.write_data_file(data_file, Nodes, Bonds, Angles, Boundary, BondTypes, model, 
                            computational_params, rest_lengths ,angle_model, angle_params)
    
    # Create folder to receive representative nets
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    
    # Initialise output array
    stress_array = [] ## for now a list
    
    # Run relaxation step ...
    print("Starting simulation for network in file %s..." %geometry_file)
    print(100 * "=")
    
    if model != '1':
        ## When hybrid bond style is used, we need to store the bond coefficients
        ## lines.
        bond_coeff_lines = NetworkClass.get_bond_coeffs(data_file)
    else:
        ## Otherwise proceed stating an empty list
        bond_coeff_lines = []
    
    run_relaxation_hybrid(dim, data_file, Boundary, model, angle_model, bond_coeff_lines)
    DN = FillerNetworkClass(data_file, "test.res","main_hybrid.in") ## Netwotk object
    cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
    stress_array.append(cauchy_stress)
    radii_deviations = DN.get_filler_radii_deviations(bond_flags, selected_nodes, filler_radius)
    angles_deviations, max_dev, min_dev = DN.get_filler_angles_deviations(angle_to_pair)
    
    # move current geometry to the corresponding folder...
    current_data_file = "Step_0.dat"
    os.system("cp %s %s" %(data_file, current_data_file))
    os.system("mv %s %s" %(current_data_file, rep_path))
    
    # Check for overlaps and spheres that might have left the domain
    is_overlaped_array = DN.any_filler_overlap(selected_nodes, filler_radius, filler_epsilon)
    if np.any(is_overlaped_array):
        print("Filler overlap occured!!")
        breakpoint()
    else:
        print("No filler overlapping detected in relaxation step.")
    initial_box = DN.get_box_lengths()
    is_missing_array = DN.any_filler_missing(selected_nodes, initial_box, filler_epsilon)
    
    if np.any(is_missing_array):
        print("There are missing fillers !!!!")
        breakpoint()
    else:
        print("Missing fillers were not detected")
    
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("max radii deviation is %g, while the min is %g" %(max(radii_deviations), min(radii_deviations)))
    print("max avg deviation (in degrees) is %g, while the min one is %g" %(max_dev, min_dev) )
    print(100 * "=")
    
    # Apply deformation history
    for i in range(1, len(stretch_array)):
        print(100 * "=")
        
        ## Run deformatio step
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main_hybrid.in')
        
        ## Check for simulation errors.
        if err:
            breakpoint()
        
        ## Reconstruct data if needed
        if model != '1':
            post.rewrite_data_file(bond_coeff_lines, data_file)
            
        ## Calculate DN information
        DN = FillerNetworkClass(data_file, "test.res","main_hybrid.in") ## Netwotk object
        F = deformation_gradient(loading, stretch_array[i])
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        radii_deviations = DN.get_filler_radii_deviations(bond_flags, selected_nodes, filler_radius)
        angles_deviations, max_dev, min_dev = DN.get_filler_angles_deviations(angle_to_pair)
        
        ## Move current data file
        current_data_file = "Step_" + str(i) + ".dat"
        os.system("cp %s %s" %(data_file, current_data_file))
        os.system("mv %s %s" %(current_data_file, rep_path))
        
        ## Check for potential overlaps and missing fillers
        is_overlaped_array = DN.any_filler_overlap(selected_nodes, filler_radius, filler_epsilon)
        if np.any(is_overlaped_array):
            print("Filler overlap occured!!")
            breakpoint()
        else:
            print("No filler overlapping detected in current deformation step.")
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("max radii deviation is %g, while the min is %g" %(max(radii_deviations), min(radii_deviations)))
        print("max avg deviation (in degrees) is %g, while the min one is %g" %(max_dev, min_dev) )
        
        ## Append current stress to the stress array
        stress_array.append(cauchy_stress)
        print(100 * "=")
        
    print("Finished simulation for network in file %s!" %geometry_file)
    print("\n\n")
    
    # Convert stress array to ndarray
    stress_array = NetworkClass.render_stress_units(np.array(stress_array), bKuhn)
    
    return stress_array



def runsim(geometry_file, model, params, dim, loading, stretch_array, 
            stretch_increment):
    """
    Run full simulation for DN with no fillers.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        
    Outputs:
        stress_array (ndarray): Array with all the stress results
    """
    
    # Unpack parameters tuple
    bKuhn, NKuhn, nub3 = params
    
    # Initialise output array
    stress_array = [] ## for now a list
    
    # Relax as generated network
    print("Starting simulation for network in file %s, with no fillers..." %geometry_file)
    print(100 * "=")
    relax_as_generated_DN(geometry_file, model, params, dim)
    DN = NetworkClass("temp.dat", "test.res", "main.in")
    computational_params = DN.get_computational_params(params) ## extract computational params
    cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
    stress_array.append(cauchy_stress)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print(100 * "=")
    
    # Apply deformation history
    for i in range(1, len(stretch_array)):
        print(100 * "=")
        ## Run deformatio step
        runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        
        ## Reconstruct data if needed
        if model != '1':
            breakpoint()
        
        ## Calculate DN information
        DN = NetworkClass("temp.dat", "test.res","main_hybrid.in") ## Netwotk object
        F = deformation_gradient(loading, stretch_array[i])
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        
        ## Append current stress to the stress array
        stress_array.append(cauchy_stress)
        print(100 * "=")
        
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    stress_array = NetworkClass.render_stress_units(np.array(stress_array), bKuhn)
    
    return stress_array



def runsim_rep(geometry_file, model, params, dim, loading, stretch_array, 
                stretch_increment, folder_names):
    """
    Run simulation for representative network without  fillers.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        folder_names (tuple): sequence of strings containing the path to place the 
                              data file.
        
    Outputs:
        stress_array (ndarray): Array with all the stress results
    """
    
    # Unpack parameters tuple
    bKuhn, NKuhn, nub3 = params
    
    # Create folder to receive representative nets
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    
    # Initialise output array
    stress_array = [] ## for now a list
    
    # Relax as generated network
    print("Starting simulation for network in file %s, with no fillers..." %geometry_file)
    print(100 * "=")
    relax_as_generated_DN(geometry_file, model, params, dim)
    DN = NetworkClass("temp.dat", "test.res", "main.in")
    computational_params = DN.get_computational_params(params) ## extract computational params
    cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
    stress_array.append(cauchy_stress)
    
    
    # move current geometry to the corresponding folder...
    current_data_file = "Step_0.dat"
    os.system("cp temp.dat %s" %current_data_file)
    os.system("mv %s %s" %(current_data_file, rep_path))
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print(100 * "=")
    
    # Apply deformation history
    for i in range(1, len(stretch_array)):
        print(100 * "=")
        ## Run deformatio step
        runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        
        
        ## Calculate DN information
        DN = NetworkClass("temp.dat", "test.res","main_hybrid.in") ## Netwotk object
        F = deformation_gradient(loading, stretch_array[i])
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        
        ## Move current data file
        current_data_file = "Step_" + str(i) + ".dat"
        os.system("cp temp.dat %s" %current_data_file)
        os.system("mv %s %s" %(current_data_file, rep_path))
        
        
        ## Append current stress to the stress array
        stress_array.append(cauchy_stress)
        print(100 * "=")
        
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    stress_array = NetworkClass.render_stress_units(np.array(stress_array), bKuhn)
    
    return stress_array



def relax_as_generated_DN(geometry_file, model, params, dim):
    """
    Perform relaxation on as generated network
    
    Inputs: 
        geometry_file (str): path to acces as-generated network
        model (str): chain model to be used.
                    '1': Gaussian chain.
                    '2': FJC (Langevin) chain.
        params (tuple): network parameters.
        dim (int): dimention of the problem
    
    Outputs:
        None
    """
    # Import relevant modules
    #from .pre_processing import readGeometry, writePositions
    
    
    # Extract Nodes, Bonds, Boudnary and BondTypes
    Nodes, Bonds, Boundary, BondTypes = pre.readGeometry(geometry_file)
    rest_lengths = {idx: 0. for idx in Bonds.keys()} ## list of rest lengths (needed to write the data file)
    
    # Calculate computational Kuhh length
    bKuhn, NKuhn, nub3 = params
    crosslinks = len(Nodes) - len(Boundary)
    computational_bKuhn = np.power(nub3 /(2 * crosslinks), 1/3)
    computational_params = (computational_bKuhn, NKuhn)
    BondTypes = {idx: NKuhn for idx in BondTypes.keys()}
    
    # Write data file
    temp_file = "temp.dat"
    pre.writePositions(temp_file, Nodes, Bonds, Boundary, BondTypes, model, computational_params, rest_lengths)
    
    # Run relaxation
    run_relaxation(dim, temp_file, Boundary, model)
    
    return


def run_relaxation_hybrid(dim, temp_file, Boundary, model, angle_model, bond_coeff_lines):
    """
    Relax network with hybrid bond styles. It works as well when all bonds
    are of the same style, i.e., harmonic.
    """
    # Generate input files for LAMMPS
    mainfile = 'main_hybrid.in'
    posfile = temp_file 
    
    # Write initial position and main file for LAMMPS
    if model == '1':
        write_main_angles(mainfile,posfile,Boundary,dim,model,angle_model)
    else:
        write_main_hybrid(mainfile, posfile, Boundary, dim, model, angle_model)
    
    
    #reference configuration: run with zero applied displacement
    err = runinc(loading = 1, inc = 0, dl = 0, dim = dim, main_file = mainfile);
    
    # If hybrid bond style was used, rewrite the bond coefficients section
    if model != '1':
        from .post_processing import rewrite_data_file
        rewrite_data_file(bond_coeff_lines, temp_file)
    
    return


def run_relaxation_angles(dim, temp_file, Boundary, model, angle_model):
    """
        This code runs the relaxation considering angle hinderence 
    """
    # Generate input files for LAMMPS
    mainfile = 'main_angles.in'
    posfile = temp_file 
    
    # Write initial position and main file for LAMMPS
    write_main_angles(mainfile,posfile,Boundary,dim,model,angle_model)
    
    
    #reference configuration: run with zero applied displacement
    err = runinc(loading = 1, inc = 0, dl = 0, dim = dim, main_file = mainfile);
    
    return


def run_relaxation(dim, temp_file, Boundary, model):
    """
        This code runs the relaxation after a certain amount of chain have been degraded 
    """
    # Generate input files for LAMMPS
    mainfile = 'main.in'
    posfile = temp_file 
    
    # Write initial position and main file for LAMMPS
    writeMain(mainfile,posfile,Boundary,dim,model)
    
    #reference configuration: run with zero applied displacement
    err = runinc(loading = 1, inc = 0, dl = 0, dim = dim, main_file = mainfile);
    
    return

def runinc(loading,inc,dl,dim, main_file, periodic_flag = False):
    """ 
    Run one deformatio increment on the DN using LAMMPS
        
    Inputs:
        loading (int): type of loading.
            1: uniaxial tension
            2: biaxial tension
            3: pure shear
        inc (int): increment number
        dl (float): stretch increment
        dim (int): problem dimension.
        main_file (str): name of lammps input file.
        periodic_flag (bool, optional): flag for periodic BC. Default is false.
        
    Outputs:
        None
    """
    #print('###Inc %d' %inc)

    #displacement increment applied to the box of dimension 1.2 if not PBC
    if periodic_flag:
        du = dl
    else:
        du = 1.2*dl
    
    #copy main file in temporary file 
    os.system('cp %s main_tmp.in' %main_file)

    if dim==3:
    
        #uniaxial loading
        if loading == 1:
            newline = "fix 1 all deform 1 x delta " + str(-du/2.) + " " + str(du/2)  + " y volume z volume remap x units box\n"

        #equi-biaxial loading
        elif loading == 2:
            newline = "fix 1 all deform 1 x delta " + str(-du/2.) + " " + str(du/2) + " y delta " + str(-du/2) + " " + str(du/2) + " z volume remap x units box\n"

        #pure shear
        elif loading == 3:
            newline = "fix 1 all deform 1 x delta " + str(-du/2) + " " + str(du/2) + " y volume z delta 0 0 remap x units box\n"
        
        
        else:
            print('invalid loading in runinc: %d' %loading)
            exit()

    #2D simulation
    else:
        newline = "fix 1 all deform 1 x delta " + str(-du/2) + " " + str(du/2) + " y volume remap x units box\n"

    #rewrite the main file by replacing the line with the loading
    fin = open('main_tmp.in','r')
    fout = open(main_file,'w')

    for line in fin:
        
        if 'delta' in line:
            fout.write(newline)
        else:
            fout.write(line)

    fin.close()
    fout.close()

    #run lammps and store result file
    os.system('~/.local/bin/lmp -in %s > log' %main_file)

    #check for error in log file
    err = checkerror('log.lammps')

    return err



def write_main_hybrid(simfile,posfile, Boundary,dim,model, angle_model, periodic_flag = False):

    """ 
    Write the main input file for LAMMPS when more that one bond style in present
    """

    min_algo='fire' #algorithm for minimization
    dmax = 0.05      #how much a single atom can move during line search
    #dmax = 0.1    # Only of very small it makes a difference
    #dmax = 10
    
    
    # Find bond style to be used
    bond_style, err = get_bond_style(model)
    if err:
        exit()
    
    # Find angle style to be used
    angle_style = get_angle_style(angle_model) 
    
    # Open file and start writting process
    f = open(simfile,'w')


    f.write('#Main input file for LAMMPS\n')

    f.write('units\tlj\n')
    f.write('dimension\t%d\n' %dim)
    if dim == 3:
        f.write('boundary\tf f f \n')
    else:
        if not periodic_flag:
            f.write('boundary\tf f p\n')
        else:
            f.write('boundary\tp p p\n')
    
    f.write('atom_style\tmolecular\n')
    f.write('bond_style\t hybrid harmonic %s\n' %(bond_style)) ## harmonic style for sphere bonds
    f.write('angle_style\t%s\n' %(angle_style))
    f.write('atom_modify\tsort 0 0\n')
    f.write('pair_style\tnone\n\n')

    f.write('read_data\t%s\n\n' %posfile)

    f.write('reset_timestep\t0\n')
    f.write('timestep\t0.0001\n')
    f.write('neighbor\t0.1 nsq\n') ## might need to be adjusted for PBC
    f.write('thermo\t1\n')
    if dim == 3:
        f.write('thermo_style\tcustom etotal press pxx pyy pzz pxy pxz pyz\n')
    else:
        f.write('thermo_style\tcustom etotal press pxx pyy pxy\n')
    
    f.write('min_style\t%s\n' %(min_algo))
    f.write('min_modify\tdmax %s\n\n' %(dmax))
    
    if not periodic_flag:
        f.write('group\tboundary id ')
        for i in range(len(Boundary)):
            f.write('%s ' %(Boundary[i]))
        f.write('\n\n')

    # Step 1: deform the box affinely 
    #delta values: change in box boundaries at the end of run  
    #Note: actual mode of deformation applied here is not important as these lines will be replaced
    #by run.py on the go
    if dim == 3:
        f.write('fix 1 all deform 1 x delta 0 0 y volume z volume remap x units box\n')
    else:
        if not periodic_flag:
            f.write('fix 1 all deform 1 x delta 0 0 y volume remap x units box\n')
        else:
            f.write('fix 1 all deform 1 x delta 0 0 y volume remap x units box\n')
        

    #need a run to apply the fix deform command above
    f.write('run 1\n\n')

    # Step 2: Apply zero force on boundary nodes (prevent their motion) and minimize energy
    if not periodic_flag:
        f.write('fix\t2 boundary setforce 0 0 0\n')
    f.write('minimize\t1e-10 1e-10 100000 10000\n\n')
    #f.write('minimize\t0 1e-16 1000 10000\n\n')

    # Step 3: remove the zero-force constraint on the boundary
    if not periodic_flag:
        f.write('unfix 2\n\n')

    # Define computation to calculate forces
    if dim == 3:
        f.write('compute\t1 boundary property/atom fx fy fz\n')
        f.write('dump\t1 boundary custom 1 test.res id type x y z c_1[1] c_1[2] c_1[3]\n')

    else:
        if not periodic_flag:
            f.write('compute\t1 boundary property/atom fx fy\n')
            f.write('dump\t1 boundary custom 1 test.res id type x y c_1[1] c_1[2]\n')
            
            f.write('dump_modify\t1 sort id\n')

    #run dummy step (0 increment) to perform the dump operation and write test.res
    f.write('run\t0\n\n')   
    
    #write new atom positions
    f.write('write_data\t%s\n\n' %posfile)

    f.close()
    
    
    
    return




def write_main_angles(simfile,posfile, Boundary,dim,model, angle_model, periodic_flag = False):

    """ 
    Write the main input file for LAMMPS
    """

    min_algo='fire' #algorithm for minimization
    dmax = 0.05      #how much a single atom can move during line search
    #dmax = 0.1    # Only of very small it makes a difference
    #dmax = 10
    
    
    # Find bond style to be used
    bond_style, err = get_bond_style(model)
    if err:
        exit()
    
    # Find angle style to be used
    angle_style = get_angle_style(angle_model) 
    
    # Open file and start writting process
    f = open(simfile,'w')


    f.write('#Main input file for LAMMPS\n')

    f.write('units\tlj\n')
    f.write('dimension\t%d\n' %dim)
    if dim == 3:
        f.write('boundary\tf f f \n')
    else:
        if not periodic_flag:
            f.write('boundary\tf f p\n')
        else:
            f.write('boundary\tp p p\n')
    
    f.write('atom_style\tmolecular\n')
    f.write('bond_style\t%s\n' %(bond_style))
    f.write('angle_style\t%s\n' %(angle_style))
    f.write('atom_modify\tsort 0 0\n')
    f.write('pair_style\tnone\n\n')

    f.write('read_data\t%s\n\n' %posfile)

    f.write('reset_timestep\t0\n')
    f.write('timestep\t0.0001\n')
    f.write('neighbor\t0.1 nsq\n') ## might need to be adjusted for PBC
    f.write('thermo\t1\n')
    if dim == 3:
        f.write('thermo_style\tcustom etotal press pxx pyy pzz pxy pxz pyz\n')
    else:
        f.write('thermo_style\tcustom etotal press pxx pyy pxy\n')
    
    f.write('min_style\t%s\n' %(min_algo))
    f.write('min_modify\tdmax %s\n\n' %(dmax))
    
    if not periodic_flag:
        f.write('group\tboundary id ')
        for i in range(len(Boundary)):
            f.write('%s ' %(Boundary[i]))
        f.write('\n\n')

    # Step 1: deform the box affinely 
    #delta values: change in box boundaries at the end of run  
    #Note: actual mode of deformation applied here is not important as these lines will be replaced
    #by run.py on the go
    if dim == 3:
        f.write('fix 1 all deform 1 x delta 0 0 y volume z volume remap x units box\n')
    else:
        if not periodic_flag:
            f.write('fix 1 all deform 1 x delta 0 0 y volume remap x units box\n')
        else:
            f.write('fix 1 all deform 1 x delta 0 0 y volume remap x units box\n')
        

    #need a run to apply the fix deform command above
    f.write('run 1\n\n')

    # Step 2: Apply zero force on boundary nodes (prevent their motion) and minimize energy
    if not periodic_flag:
        f.write('fix\t2 boundary setforce 0 0 0\n')
    f.write('minimize\t1e-10 1e-10 100000 10000\n\n')
    #f.write('minimize\t0 1e-16 1000 10000\n\n')

    # Step 3: remove the zero-force constraint on the boundary
    if not periodic_flag:
        f.write('unfix 2\n\n')

    # Define computation to calculate forces
    if dim == 3:
        f.write('compute\t1 boundary property/atom fx fy fz\n')
        f.write('dump\t1 boundary custom 1 test.res id type x y z c_1[1] c_1[2] c_1[3]\n')

    else:
        if not periodic_flag:
            f.write('compute\t1 boundary property/atom fx fy\n')
            f.write('dump\t1 boundary custom 1 test.res id type x y c_1[1] c_1[2]\n')
            
            f.write('dump_modify\t1 sort id\n')

    #run dummy step (0 increment) to perform the dump operation and write test.res
    f.write('run\t0\n\n')   
    
    #write new atom positions
    f.write('write_data\t%s\n\n' %posfile)

    f.close()
    
    
    
    return


def writeMain(simfile,posfile,Boundary,dim,model, periodic_flag = False):

    """ 
    Write the main input file for LAMMPS
    """

    min_algo='fire' #algorithm for minimization
    dmax = 0.05      #how much a single atom can move during line search
    #dmax = 0.1    # Only of very small it makes a difference
    #dmax = 10
    
    
    # Find bond style to be used
    bond_style, err = get_bond_style(model)
    if err:
        exit()
    
    f = open(simfile,'w')


    f.write('#Main input file for LAMMPS\n')

    f.write('units\tlj\n')
    f.write('dimension\t%d\n' %dim)
    if dim == 3:
        f.write('boundary\tf f f \n')
    else:
        if not periodic_flag:
            f.write('boundary\tf f p\n')
        else:
            f.write('boundary\tp p p\n')
    
    f.write('atom_style\tbond\n')
    f.write('bond_style\t%s\n' %(bond_style))
    f.write('atom_modify\tsort 0 0\n')
    f.write('pair_style\tnone\n\n')

    f.write('read_data\t%s\n\n' %posfile)

    f.write('reset_timestep\t0\n')
    f.write('timestep\t0.0001\n')
    f.write('neighbor\t0.1 nsq\n') ## might need to be adjusted for PBC
    f.write('thermo\t1\n')
    if dim == 3:
        f.write('thermo_style\tcustom etotal press pxx pyy pzz pxy pxz pyz\n')
    else:
        f.write('thermo_style\tcustom etotal press pxx pyy pxy\n')
    
    f.write('min_style\t%s\n' %(min_algo))
    f.write('min_modify\tdmax %s\n\n' %(dmax))
    
    if not periodic_flag:
        f.write('group\tboundary id ')
        for i in range(len(Boundary)):
            f.write('%s ' %(Boundary[i]))
        f.write('\n\n')

    # Step 1: deform the box affinely 
    #delta values: change in box boundaries at the end of run  
    #Note: actual mode of deformation applied here is not important as these lines will be replaced
    #by run.py on the go
    if dim == 3:
        f.write('fix 1 all deform 1 x delta 0 0 y volume z volume remap x units box\n')
    else:
        if not periodic_flag:
            f.write('fix 1 all deform 1 x delta 0 0 y volume remap x units box\n')
        else:
            f.write('fix 1 all deform 1 x delta 0 0 y volume remap x units box\n')
        

    #need a run to apply the fix deform command above
    f.write('run 1\n\n')

    # Step 2: Apply zero force on boundary nodes (prevent their motion) and minimize energy
    if not periodic_flag:
        f.write('fix\t2 boundary setforce 0 0 0\n')
    f.write('minimize\t1e-12 1e-12 1000 10000\n\n')
    #f.write('minimize\t0 1e-16 1000 10000\n\n')

    # Step 3: remove the zero-force constraint on the boundary
    if not periodic_flag:
        f.write('unfix 2\n\n')

    # Define computation to calculate forces
    if dim == 3:
        f.write('compute\t1 boundary property/atom fx fy fz\n')
        f.write('dump\t1 boundary custom 1 test.res id type x y z c_1[1] c_1[2] c_1[3]\n')

    else:
        if not periodic_flag:
            f.write('compute\t1 boundary property/atom fx fy\n')
            f.write('dump\t1 boundary custom 1 test.res id type x y c_1[1] c_1[2]\n')
            
            f.write('dump_modify\t1 sort id\n')

    #run dummy step (0 increment) to perform the dump operation and write test.res
    f.write('run\t0\n\n')   
    
    #write new atom positions
    f.write('write_data\t%s\n\n' %posfile)

    f.close()

    return


def get_bond_style(model):
    """
    Get string identifier within lammps of the chain model used
    
    Inputs:
        model (str): string informing the chain model
        
    Outputs:
        bond_style (str): string identifier of the bond style in lammps.
        err (bool): flag informing if model passed to the function is valid.
    """
    err = False;
    if model in ['1', '5']:
        bond_style = 'harmonic'
        
    elif model == '2':
        bond_style = 'langevin'
        
    elif model == '3':
        bond_style = 'Xlangevin'
        
    elif model == '4':
        bond_style = 'Fraclangevin';
        
    elif model == '6':
        bond_style = 'Fracharmonic';
        
    else:
        print('unknown bond type: %s' %model)
        err = True
        exit()
    
    return bond_style, err

def get_angle_style(model):
    """
    Get string identifier within lammps of the chain model used
    
    Inputs:
        model (str): string informing the angle potential type
        
    Outputs:
        angle_style (str): string identifier of the angle style in lammps.
        
    """
    if model == '1':
        angle_style = 'harmonic'
    return angle_style


def checkerror(filename):
    
    fin = open(filename,'r')

    for line in fin:
        
        if 'ERROR' in line:
            print('Error message in log.lammps:')
            print(line)
            return True
    return False




if __name__ == "__main__":
    main()