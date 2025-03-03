import numpy as np
import os
import utils.pre_processing as pre
import utils.post_processing as post
from .network_class import NetworkClass, FracNetworkClass
from .loading import *
from pathlib import Path


def runsim(geometry_file, model, params, dim, loading, stretch_increments, data_file):
    """
    Run full simulation for DN with no fillers.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_increment (float):
        
    Outputs:
        stress_array (ndarray): Array with all the stress results
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    bKuhn, NKuhn, nub3 = params
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] ## for now a list
    nominal_stress_array = [] ## for now a list
    
    # Relax as generated network
    print("Starting simulation for network in file %s, with no fillers..." %geometry_file)
    print(100 * "=")
    relax_as_generated_DN(geometry_file, model, params, dim, data_file)
    
    # Create initial DN object
    DN_initial = NetworkClass(data_file, "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = NetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print(100 * "=")
    
    # Apply deformation history
    for i, stretch_increment in enumerate(stretch_increments):
        print(100 * "=")
        ## Run deformatio step
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        
        if err:
            print("Locking issues with chains, trying, with reduces increment")
            break
        else:
            F = deformation_gradient(loading, stretch_array[-1] + stretch_increment)
            stretch_array.append(F[0])
            DN = NetworkClass(data_file, "test.res","main.in") ## Netwotk object
            
        ## Calculate stresses
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        nominal_stress = NetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
        
        ## Append current stress to the stress array
        cauchy_stress_array.append(cauchy_stress)
        nominal_stress_array.append(nominal_stress)
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
        
        ## Append current stress to the stress array
        print(100 * "=")
        
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    cauchy_stress_array = NetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = NetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    
    out = np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array)
    
    return out



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



def runsim_frac(geometry_file, model, params, dim, loading, stretch_increment, 
                    failure_criterion, data_file):
    """
    Run full simulation for DN where on-and-off scissions are allowed to 
    happen. We run the simulation until failure is detected
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        failure_criterion (int): 
        
    Outputs:
        stress_array (ndarray): Array with all the stress results
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, NKuhn, nub3, critical_r_Nb = params
    
    # Initialise output arrays and counters
    stretch_array = []
    cauchy_stress_array = []
    nominal_stress_array = []
    fraction_broken_chains = []
    G_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s, with no fillers..." %geometry_file)
    print(100 * "=")
    relax_as_generated_DN(geometry_file, model, params, dim, data_file)
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = NetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds) 
    nBonds_beginning = initial_nBonds
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # Calculare the shear modulus
    G = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn)
    G_array.append(G)
    
    # Query for initial failure
    path_exists = DN_initial.any_path(failure_criterion, initial_Nodes)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print("G = %g kPa" %G)
    if path_exists:
        print("No connectivity issues found in the reference configuration, proceed")
        fraction_broken_chains.append(0.)
    else:
        print("Connectivity problems found, aborting simulation")
    print(100 * "=")
    
    # Apply deformation until failure is detected
    while path_exists:
        print(100 * "=")
        ## Run deformatio step
        i += 1
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        
        ## Check if simulation was aborted
        if err:
            print("Increment failed")
            i -= 1
            ## Run reduced increments
            err = run_reduced_inc(data_file, stretch_increment, loading, dim, main_file = 'main.in')
            
            if err:
                print("Reduced increments did not work..")
                break
            else:
                print("Reduced increments worked!")
                i +=1
        
        ## Calculate current deformatio gradient
        F = deformation_gradient(loading, stretch_array[i - 1] + stretch_increment)
        stretch_array.append(F[0])
        
        ## Initialise DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
        
        ## Check if scissions ocurred, and if yes, relax the network
        nBonds = len(DN.get_nodes_and_bonds()[1])
        scission_detected = nBonds < nBonds_beginning
        if scission_detected:
            nBonds = relax_unitl_no_scissions(data_file, dim, "main.in", nBonds_beginning)
            nBonds_beginning = nBonds
        
        ## Update DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
        
        ## Calculate stresses
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        nominal_stress = NetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
        
        ## Calculate the damaged shear modulus
        if scission_detected:
            G = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn)
        
        ## Append current stress to the stress array
        cauchy_stress_array.append(cauchy_stress)
        nominal_stress_array.append(nominal_stress)
        G_array.append(G)
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
        print("G = %g kPa" %G)
        
        ## Assess if failure occurred
        nBonds = len(DN.get_nodes_and_bonds()[1]) 
        fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
        path_exists = DN.any_path(failure_criterion, initial_Nodes)
        if path_exists:
            print("Failure not detected. Move to next increment")
            print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
        else:
            print("Failure detected at increment %d" %i)
            print("Failure occurred with %g percent of broken chains" %(100 * fraction_broken_chains[-1]))
            print("Breaking simulation")
            break
        
        print(100 * "=")
        
    
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    cauchy_stress_array = NetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = NetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    
    out = (np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array), 
            np.array(fraction_broken_chains), np.array(G_array)
            )
    
    return out




def runsim_frac_rep(geometry_file, model, params, dim, loading, stretch_increment, 
                    failure_criterion, data_file, folder_names):
    """
    Run full simulation for DN where on-and-off scissions are allowed to 
    happen. We run the simulation until failure is detected. 
    
    NOTE: this function is used to representative simulations only
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        failure_criterion (int): 
        folder_names (tuple): string to form path where geometries will be placed
    
    Outputs:
        out (tuple): results of the simulation
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, NKuhn, nub3, critical_r_Nb = params
    
    # Creat folde to receive
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] ## for now a list
    nominal_stress_array = [] ## for now a list
    fraction_broken_chains = []
    G_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print(100 * "=")
    relax_as_generated_DN(geometry_file, model, params, dim, data_file)
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get pre-stretch of the network
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # Calculare the shear modulus
    G = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn)
    G_array.append(G)
    
    # Query for initial failure
    path_exists = DN_initial.any_path(failure_criterion, initial_Nodes)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print("G = %g kPa" %G)
    if path_exists:
        print("No connectivity issues found in the reference configuration, proceed")
        fraction_broken_chains.append(0.)
    else:
        print("Connectivity problems found, aborting simulation")
        
    print(100 * "=")
    
    # move relaxed geometry to the representative folder
    current_data_file = "Step_0.dat"
    os.system("cp %s %s" %(data_file, current_data_file))
    os.system("mv %s %s" %(current_data_file, rep_path))
    
    # Apply deformation until failure is detected
    while path_exists:
        print(100 * "=")
        ## Update the number of chains at start of step
        
        ## Run deformatio step
        i += 1
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        
        ## Check if simulation was aborted
        if err:
            print("Increment failed")
            i -= 1
            ## Run reduced increments
            err = run_reduced_inc(data_file, stretch_increment, loading, dim, main_file = 'main.in')
            
            if err:
                print("Reduced increments did not work..")
                break
            else:
                print("Reduced increments worked!")
                i +=1
            
        
        ## Calculate current deformatio gradient
        F = deformation_gradient(loading, stretch_array[i - 1] + stretch_increment)
        stretch_array.append(F[0])
        
        ## Initialise DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
        
        ## Check if scissions ocurred, and if yes, relax the network
        nBonds = len(DN.get_nodes_and_bonds()[1])
        scission_detected = nBonds < nBonds_beginning
        if scission_detected:
            nBonds = relax_unitl_no_scissions(data_file, dim, "main.in", nBonds_beginning)
            nBonds_beginning = nBonds
            
        ## Update DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
        
        ## Calculate stresses
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
        
        ## Calculate the damaged shear modulus if scissions were detected
        if scission_detected:
            G = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn)
        
        ## Append current stress to the stress array
        cauchy_stress_array.append(cauchy_stress)
        nominal_stress_array.append(nominal_stress)
        G_array.append(G)
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
        print("G = %g kPa" %G)
        
        ## Move geometry to representative folder
        current_data_file = "Step_" + str(i) + ".dat"
        os.system("cp %s %s" %(data_file, current_data_file))
        os.system("mv %s %s" %(current_data_file, rep_path))
        
        ## Asses if failure occurred
        nBonds = len(DN.get_nodes_and_bonds()[1]) 
        fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
        path_exists = DN.any_path(failure_criterion, initial_Nodes)
        if path_exists:
            print("Failure not detected. Move to next increment")
            print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
        else:
            print("Failure detected at increment %d" %i)
            print("Failure occurred with %g percent of broken chains" %(100 * fraction_broken_chains[-1]))
            print("Breaking simulation")
            break
        
        print(100 * "=")
        
    
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    cauchy_stress_array = FracNetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = FracNetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    
    out = (np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array), 
                np.array(fraction_broken_chains), np.array(G_array) , preStretch_distr
           )
    
    return out




def runsim_frac_multi(geometry_file, model, params, dim, loading, stretch_increment, 
                            failure_criterion, data_file, strengths, strong_fraction):
    """
    Run full simulation for DN where on-and-off scissions are allowed to 
    happen. We run the simulation until failure is detected. This function
    was desgined for cases where there is a bimodal distribution of chain
    strengths in the network.
    
    NOTE: this function is used to representative simulations only.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        failure_criterion (int): 
        strengths (tuple): weak and strong chains strengths.
        strong_fraction (float): fraction of strong chains in the network.
    
    Outputs:
        out (tuple): results of the simulation
        Wf (float): work of fracture
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, NKuhn, nub3 = params 
        
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] ## for now a list
    nominal_stress_array = [] ## for now a list
    fraction_broken_chains = []
    G_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print(100 * "=")
    relax_multi(geometry_file, model, params, dim, data_file, strengths, strong_fraction)
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = NetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # Calculare the shear modulus
    G = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn)
    G_array.append(G)
    
    # Query for initial failure
    path_exists = DN_initial.any_path(failure_criterion, initial_Nodes)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print("G = %g kPa" %G)
    if path_exists:
        print("No connectivity issues found in the reference configuration, proceed")
        fraction_broken_chains.append(0.)
    else:
        print("Connectivity problems found, aborting simulation")
        
    print(100 * "=")
    
    
    # Apply deformation until failure is detected
    while path_exists:
        print(100 * "=")
        ## Run deformatio step
        i += 1
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        
        ## Check if simulation was aborted
        if err:
            print("Increment failed")
            i -= 1
            ## Run reduced increments
            err = run_reduced_inc(data_file, stretch_increment, loading, dim, main_file = 'main.in')
            
            if err:
                print("Reduced increments did not work..")
                break
            else:
                print("Reduced increments worked!")
                i += 1 ## update increment number
            
        
        ## Calculate current deformatio gradient
        F = deformation_gradient(loading, stretch_array[i - 1] + stretch_increment)
        stretch_array.append(F[0])
        
        ## Initialise DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
        
        ## Check if scissions ocurred, and if yes, relax the network
        nBonds = len(DN.get_nodes_and_bonds()[1])
        scission_detected = nBonds < nBonds_beginning
        if scission_detected:
            nBonds = relax_unitl_no_scissions(data_file, dim, "main.in", nBonds_beginning)
            nBonds_beginning = nBonds
        
        ## Update DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
        
        ## Calculate stresses
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        nominal_stress = NetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
        
        ## Calculate the damaged shear modulus
        if scission_detected:
            G = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn, loading)
            
        
        ## Append current stress to the stress array
        cauchy_stress_array.append(cauchy_stress)
        nominal_stress_array.append(nominal_stress)
        G_array.append(G)
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
        print("G = %g kPa" %G)
        
        
        ## Assess if failure occurred
        nBonds = len(DN.get_nodes_and_bonds()[1]) 
        fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
        path_exists = DN.any_path(failure_criterion, initial_Nodes)
        if path_exists:
            print("Failure not detected. Move to next increment")
            print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
        else:
            print("Failure detected at increment %d" %i)
            print("Failure occurred with %g percent of broken chains" %(100 * fraction_broken_chains[-1]))
            print("Breaking simulation")
            break
        
        print(100 * "=")
        
    
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    cauchy_stress_array = NetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = NetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    out = (np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array), 
                np.array(fraction_broken_chains), np.array(G_array)
           )
        
    return out






def runsim_frac_multi_rep(geometry_file, model, params, dim, loading, stretch_increment, 
                            failure_criterion, data_file, strengths, strong_fraction, 
                            folder_names):
    """
    Run full simulation for DN where on-and-off scissions are allowed to 
    happen. We run the simulation until failure is detected. This function
    was desgined for cases where there is a bimodal distribution of chain
    strengths in the network.
    
    NOTE: this function is used to representative simulations only.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        failure_criterion (int): 
        strengths (tuple): weak and strong chains strengths.
        strong_fraction (float): fraction of strong chains in the network.
        folder_names (tuple): string to form path where geometries will be placed
    
    Outputs:
        out (tuple): results of the simulations
        
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, NKuhn, nub3 = params 
        
    
    # Creat folde to receive
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] ## for now a list
    nominal_stress_array = [] ## for now a list
    fraction_broken_chains = []
    G_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print(100 * "=")
    relax_multi(geometry_file, model, params, dim, data_file, strengths, strong_fraction)
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = NetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get pre-stretch of the network
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # Calculare the shear modulus
    G = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
    G_array.append(G)
    
    # Query for initial failure
    path_exists = DN_initial.any_path(failure_criterion, initial_Nodes)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print("G = %g kPa" %G)
    if path_exists:
        print("No connectivity issues found in the reference configuration, proceed")
        fraction_broken_chains.append(0.)
    else:
        print("Connectivity problems found, aborting simulation")
        
    print(100 * "=")
    
    # move relaxed geometry to the representative folder
    current_data_file = "Step_0.dat"
    os.system("cp %s %s" %(data_file, current_data_file))
    os.system("mv %s %s" %(current_data_file, rep_path))
    
    # Apply deformation until failure is detected
    while path_exists:
        print(100 * "=")
        ## Run deformatio step
        i += 1
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        
        ## Check if simulation was aborted
        if err:
            print("Increment failed")
            i -= 1
            ## Run reduced increments
            err = run_reduced_inc(data_file, stretch_increment, loading, dim, main_file = 'main.in')
            
            if err:
                print("Reduced increments did not work..")
                break
            else:
                print("Reduced increments worked!")
                i += 1 ## update increment number
            
        
        ## Calculate current deformatio gradient
        F = deformation_gradient(loading, stretch_array[i - 1] + stretch_increment)
        stretch_array.append(F[0])
        
        ## Initialise DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
        
        ## Check if scissions ocurred, and if yes, relax the network
        nBonds = len(DN.get_nodes_and_bonds()[1])
        scission_detected = nBonds < nBonds_beginning
        if scission_detected:
            nBonds = relax_unitl_no_scissions(data_file, dim, "main.in", nBonds_beginning)
            nBonds_beginning = nBonds
        
        ## Update DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
        
        ## Calculate stresses
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        nominal_stress = NetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
        
        ## Calculate the damaged shear modulus
        if scission_detected:
            G = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn, loading)
            
        
        ## Append current stress to the stress array
        cauchy_stress_array.append(cauchy_stress)
        nominal_stress_array.append(nominal_stress)
        G_array.append(G)
        
        ## Print currrent step data
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
        print("G = %g kPa" %G)
        
        ## Move geometry to representative folder
        current_data_file = "Step_" + str(i) + ".dat"
        os.system("cp %s %s" %(data_file, current_data_file))
        os.system("mv %s %s" %(current_data_file, rep_path))
        
        ## Assess if failure occurred
        nBonds = len(DN.get_nodes_and_bonds()[1]) 
        fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
        path_exists = DN.any_path(failure_criterion, initial_Nodes)
        if path_exists:
            print("Failure not detected. Move to next increment")
            print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
        else:
            print("Failure detected at increment %d" %i)
            print("Failure occurred with %g percent of broken chains" %(100 * fraction_broken_chains[-1]))
            print("Breaking simulation")
            break
        
        print(100 * "=")
        
    
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    cauchy_stress_array = NetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = NetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    out = (np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array), 
                np.array(fraction_broken_chains), np.array(G_array), preStretch_distr
           )
        
    return out





def runsim_cyclic_rep(geometry_file, model, params, dim, loading, stretch_increment, 
                    peak_stretches, failure_criterion, data_file, 
                    folder_names):
    """
    Run cyclic simulation.  
    
    NOTE: this function is used to representative simulations only
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        failure_criterion (int): 
        folder_names (tuple): string to form path where geometries will be placed
    
    Outputs:
        out (tuple): results of the simulation
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, NKuhn, nub3, critical_r_Nb = params
    
    # Creat folde to receive
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] ## for now a list
    nominal_stress_array = [] ## for now a list
    fraction_broken_chains = []
    G_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print(100 * "=")
    relax_as_generated_DN(geometry_file, model, params, dim, data_file)
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get pre-stretch of the network
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # Calculare the shear modulus
    G = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
    G_array.append(G)
    
    # Query for initial failure
    path_exists = DN_initial.any_path(failure_criterion, initial_Nodes)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print("G = %g kPa" %G)
    if path_exists:
        print("No connectivity issues found in the reference configuration, proceed")
        fraction_broken_chains.append(0.)
    else:
        print("Connectivity problems found, aborting simulation")
        
    print(100 * "=")
    
    # move relaxed geometry to the representative folder
    current_data_file = "Step_0.dat"
    os.system("cp %s %s" %(data_file, current_data_file))
    os.system("mv %s %s" %(current_data_file, rep_path))
    
    # Apply deformation until failure is detected
    reverse_flag = False ## reverse load flag
    
    
    for peak in peak_stretches:
        ## Print what is the target peak stretch
        print("Current peak stretch: %g" %peak)
        
        ## Reset flag informing if loop has stabilised, and other variables
        is_stable = False
        is_peak = False 
        peak_stress = 0.
        current_stretch_increment = stretch_increment ## variable to store the stretch increment size
        
        while not is_stable:
            
            print(100 * "=")
            ## Run deformatio step
            i += 1
            err = runinc(loading, i + 1, current_stretch_increment, dim, main_file = 'main.in')
            
            ## Check if simulation was aborted
            if err:
                print("Increment failed")
                i -= 1
                ## Run reduced increments
                err = run_reduced_inc(data_file, current_stretch_increment, loading, dim, main_file = 'main.in')
                
                if err:
                    print("Reduced increments did not work..")
                    break
                else:
                    print("Reduced increments worked!")
                    i +=1
                
            
            ## Calculate current deformatio gradient and check if peak was reached
            F = deformation_gradient(loading, stretch_array[i - 1] + current_stretch_increment)
            stretch_array.append(stretch_array[i - 1] + current_stretch_increment)
            
            
            ## Initialise DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
            
            ## Check if scissions ocurred, and if yes, relax the network
            nBonds = len(DN.get_nodes_and_bonds()[1])
            scission_detected = nBonds < nBonds_beginning
            if scission_detected:
                nBonds = relax_unitl_no_scissions(data_file, dim, "main.in", nBonds_beginning)
                nBonds_beginning = nBonds
                
            ## Update DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
            
            ## Calculate stresses
            cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
            nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
            
            ## Calculate the damaged shear modulus if scissions were detected
            if scission_detected:
                G = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn, loading)
            
            ## Append current stress to the stress array
            cauchy_stress_array.append(cauchy_stress)
            nominal_stress_array.append(nominal_stress)
            G_array.append(G)
            
            ## Print currrent step data
            print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
            print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
            print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
            print("G = %g kPa" %G)
            
            ## Move geometry to representative folder
            current_data_file = "Step_" + str(i) + ".dat"
            os.system("cp %s %s" %(data_file, current_data_file))
            os.system("mv %s %s" %(current_data_file, rep_path))
            
            ## Check at which stage of the loading we are
            current_stretch_increment, is_peak = reverse_load(peak, stretch_array[-1], current_stretch_increment)
            
            ## If peak was reached, assess if current loop is stabilised
            if is_peak:
                is_stable, peak_stress = check_loop_stability(peak_stress, max(cauchy_stress))
                is_peak = False ## set to false again to the peak stress is not checked again
            
            ## Asses if failure occurred
            nBonds = len(DN.get_nodes_and_bonds()[1]) 
            fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
            path_exists = DN.any_path(failure_criterion, initial_Nodes)
            
            if path_exists:
                print("Failure not detected. Move to next increment")
                print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
            else:
                print("Failure detected at increment %d" %i)
                print("Failure occurred with %g percent of broken chains" %(100 * fraction_broken_chains[-1]))
                print("Breaking simulation")
                break
            
            print(100 * "=")
        
        
    
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    cauchy_stress_array = FracNetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = FracNetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    
    out = (np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array), 
                np.array(fraction_broken_chains), np.array(G_array)
           )
    
    return out




def runsim_cyclic_multi_rep(geometry_file, model, params, dim, loading, stretch_increment, 
                            peak_stretches, failure_criterion, data_file, strengths, 
                            strong_fraction, folder_names):
    """
    Run cyclic simulations for the prescribed peak stetches. This function
    was desgined for cases where there is a bimodal distribution of chain
    strengths in the network.
    
    NOTE: this function is used to representative simulations only.
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        failure_criterion (int): 
        strengths (tuple): weak and strong chains strengths.
        strong_fraction (float): fraction of strong chains in the network.
        folder_names (tuple): string to form path where geometries will be placed
    
    Outputs:
        out (tuple): results of the simulations
        
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, NKuhn, nub3 = params 
        
    
    # Creat folde to receive
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] ## for now a list
    nominal_stress_array = [] ## for now a list
    fraction_broken_chains = []
    G_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print(100 * "=")
    relax_multi(geometry_file, model, params, dim, data_file, strengths, strong_fraction)
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = NetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get pre-stretch of the network
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # Calculare the shear modulus
    G = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
    G_array.append(G)
    
    # Query for initial failure
    path_exists = DN_initial.any_path(failure_criterion, initial_Nodes)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print("G = %g kPa" %G)
    if path_exists:
        print("No connectivity issues found in the reference configuration, proceed")
        fraction_broken_chains.append(0.)
    else:
        print("Connectivity problems found, aborting simulation")
        
    print(100 * "=")
    
    # move relaxed geometry to the representative folder
    current_data_file = "Step_0.dat"
    os.system("cp %s %s" %(data_file, current_data_file))
    os.system("mv %s %s" %(current_data_file, rep_path))
    
    # Apply cyclic deformation
    for peak in peak_stretches:
        ## Print what is the target peak stretch
        print("Current peak stretch: %g" %peak)
        
        ## Reset flag informing if loop has stabilised, and other variables
        is_stable = False
        is_peak = False 
        peak_stress = 0.
        current_stretch_increment = stretch_increment ## variable to store the stretch increment size
        
        while not is_stable:
            
            print(100 * "=")
            ## Run deformatio step
            i += 1
            err = runinc(loading, i + 1, current_stretch_increment, dim, main_file = 'main.in')
            
            ## Check if simulation was aborted
            if err:
                print("Increment failed")
                i -= 1
                ## Run reduced increments
                err = run_reduced_inc(data_file, stretch_increment, loading, dim, main_file = 'main.in')
                
                if err:
                    print("Reduced increments did not work..")
                    break
                else:
                    print("Reduced increments worked!")
                    i +=1
                
            
            ## Calculate current deformatio gradient and check if peak was reached
            F = deformation_gradient(loading, stretch_array[i - 1] + current_stretch_increment)
            stretch_array.append(stretch_array[i - 1] + current_stretch_increment)
            
            ## Initialise DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
            
            ## Check if scissions ocurred, and if yes, relax the network
            nBonds = len(DN.get_nodes_and_bonds()[1])
            scission_detected = nBonds < nBonds_beginning
            if scission_detected:
                nBonds = relax_unitl_no_scissions(data_file, dim, "main.in", nBonds_beginning)
                nBonds_beginning = nBonds
                
            ## Update DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
            
            ## Calculate stresses
            cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
            nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
            
            ## Calculate the damaged shear modulus if scissions were detected
            if scission_detected:
                G = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn, loading)
            
            ## Append current stress to the stress array
            cauchy_stress_array.append(cauchy_stress)
            nominal_stress_array.append(nominal_stress)
            G_array.append(G)
            
            ## Print currrent step data
            print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
            print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
            print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
            print("G = %g kPa" %G)
            
            ## Move geometry to representative folder
            if i % 10 == 0:
                current_data_file = "Step_" + str(i) + ".dat"
                os.system("cp %s %s" %(data_file, current_data_file))
                os.system("mv %s %s" %(current_data_file, rep_path))
            
            ## Check at which stage of the loading we are
            current_stretch_increment, is_peak = reverse_load(peak, stretch_array[-1], current_stretch_increment)
            
            ## If peak was reached, assess if current loop is stabilised
            if is_peak:
                is_stable, peak_stress = check_loop_stability(peak_stress, max(cauchy_stress))
                is_peak = False ## set to false again to the peak stress is not checked again
            
            ## Asses if failure occurred
            nBonds = len(DN.get_nodes_and_bonds()[1]) 
            fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
            path_exists = DN.any_path(failure_criterion, initial_Nodes)
            
            if path_exists:
                print("Failure not detected. Move to next increment")
                print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
            else:
                print("Failure detected at increment %d" %i)
                print("Failure occurred with %g percent of broken chains" %(100 * fraction_broken_chains[-1]))
                print("Breaking simulation")
                break
            
            print(100 * "=")
        
    
    print("Finished simulation for network in file %s!" %geometry_file)
    
    # Convert stress array to ndarray
    cauchy_stress_array = FracNetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = FracNetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    
    out = (np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array), 
                np.array(fraction_broken_chains), np.array(G_array)
           )
        
    return out






def runsim_cyclic_anisotropy_rep(model, params, dim, loading, stretch_increment, peak_stretches, 
                                    last_stretch, failure_criterion, data_file, folder_names):
    """
    Run cyclic simulation to probe anisitropy
    
    NOTE: this function is used to representative simulations only
    
    Inputs:
        geometry_file (str):
        model (str):
        params (tuple):
        dim (int):
        loading (int):
        stretch_array (ndarray):
        stretch_increment (float):
        peak_stretches (tuple): peak stretches to be probed
        failure_criterion (int): 
        folder_names (tuple): string to form path where geometries will be placed
    
    Outputs:
        out (tuple): results of the simulation
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, NKuhn, nub3, critical_r_Nb = params
    
    # Creat folde to receive
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] ## for now a list
    nominal_stress_array = [] ## for now a list
    fraction_broken_chains = []
    G_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting stretching in transverse direction to probe anisotropy")
    print(100 * "=")
    unload(data_file, dim, -stretch_increment, last_stretch)
    
    # Create initial DN object from the referecne prisinte DN
    DN_pristine = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    pristine_Nodes, pristine_Bonds = DN_pristine.get_nodes_and_bonds()
    pristine_nBonds = len(pristine_Bonds)
    
    # Create the initial damaged configuration
    os.system("cp %s DN_ref_dam.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref_dam.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    previous_broken = pristine_nBonds - initial_nBonds ## chains broken in the first loading
    
    
    # Append initial information
    stretch_array.append( 1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    
    # Calculare the shear modulus
    G = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
    G_array.append(G)
    
    # Query for initial failure
    path_exists = DN_initial.any_path(failure_criterion, initial_Nodes)
    
    # ... and print initial information
    print("F_11 = 1, F_22 = 1, F_33 = 1")
    print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
    print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
    print("G = %g kPa" %G)
    if path_exists:
        print("No connectivity issues found in the reference configuration, proceed")
        fraction_broken_chains.append(previous_broken / pristine_nBonds)
    else:
        print("Connectivity problems found, aborting simulation")
        
    print(100 * "=")
    
    # move relaxed geometry to the representative folder
    current_data_file = "Step_0.dat"
    os.system("cp %s %s" %(data_file, current_data_file))
    os.system("mv %s %s" %(current_data_file, rep_path))
    
    # Apply deformation until failure is detected
    reverse_flag = False ## reverse load flag
    
    for peak in peak_stretches:
        ## Print what is the target peak stretch
        print("Current peak stretch: %g" %peak)
        
        ## Reset flag informing if loop has stabilised, and other variables
        is_stable = False
        is_peak = False 
        peak_stress = 0.
        current_stretch_increment = stretch_increment ## variable to store the stretch increment size
        
        while not is_stable:
            
            print(100 * "=")
            ## Run deformatio step
            i += 1
            err = runinc(loading, i + 1, current_stretch_increment, dim, main_file = 'main.in')
            breakpoint()
            ## Check if simulation was aborted
            if err:
                print("Increment failed")
                i -= 1
                ## Run reduced increments
                err = run_reduced_inc(data_file, stretch_increment, loading, dim, main_file = 'main.in')
                
                if err:
                    print("Reduced increments did not work..")
                    break
                else:
                    print("Reduced increments worked!")
                    i +=1
                
            
            ## Calculate current deformatio gradient and check if peak was reached
            F = deformation_gradient(loading, stretch_array[i - 1] + current_stretch_increment)
            stretch_array.append(stretch_array[i - 1] + current_stretch_increment)
            
            
            ## Initialise DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
            
            ## Check if scissions ocurred, and if yes, relax the network
            nBonds = len(DN.get_nodes_and_bonds()[1])
            scission_detected = nBonds < nBonds_beginning
            if scission_detected:
                nBonds = relax_unitl_no_scissions(data_file, dim, "main.in", nBonds_beginning)
                nBonds_beginning = nBonds
                
            ## Update DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
            
            ## Calculate stresses
            cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
            nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
            
            ## Calculate the damaged shear modulus if scissions were detected
            if scission_detected:
                G = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn, loading)
            
            ## Append current stress to the stress array
            cauchy_stress_array.append(cauchy_stress)
            nominal_stress_array.append(nominal_stress)
            G_array.append(G)
            
            ## Print currrent step data
            print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
            print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
            print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
            print("G = %g kPa" %G)
            
            ## Move geometry to representative folder every 10 time-steps
            if i % 10 == 0:
                current_data_file = "Step_" + str(i) + ".dat"
                os.system("cp %s %s" %(data_file, current_data_file))
                os.system("mv %s %s" %(current_data_file, rep_path))
            
            ## Check at which stage of the loading we are
            current_stretch_increment, is_peak = reverse_load(peak, stretch_array[-1], current_stretch_increment)
            
            ## If peak was reached, assess if current loop is stabilised
            if is_peak:
                is_stable, peak_stress = check_loop_stability(peak_stress, max(cauchy_stress))
                is_peak = False ## set to false again to the peak stress is not checked again
            
            ## Asses if failure occurred
            nBonds = len(DN.get_nodes_and_bonds()[1]) 
            fraction_broken_chains.append(((initial_nBonds - nBonds)  + previous_broken)/ pristine_nBonds)
            path_exists = DN.any_path(failure_criterion, initial_Nodes)
            
            if path_exists:
                print("Failure not detected. Move to next increment")
                print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
            else:
                print("Failure detected at increment %d" %i)
                print("Failure occurred with %g percent of broken chains" %(100 * fraction_broken_chains[-1]))
                print("Breaking simulation")
                break
            
            print(100 * "=")
        
    
    print("Finished loading in transversal direction")
    
    # Convert stress array to ndarray
    cauchy_stress_array = FracNetworkClass.render_stress_units(np.array(cauchy_stress_array), bKuhn)
    nominal_stress_array = FracNetworkClass.render_stress_units(np.array(nominal_stress_array), bKuhn)
    
    out = (np.array(stretch_array), np.array(cauchy_stress_array), np.array(nominal_stress_array), 
                np.array(fraction_broken_chains), np.array(G_array)
           )
    
    return out




def unload(data_file, dim, stretch_increment, last_stretch):
    """
    Unload DN to F = I after hysteresis loop is stabilised
    """
    
    # Create DN object
    DN = FracNetworkClass(data_file, "", "")
    _, box_lengths = DN.get_box()
    
    # Based on the loading based on the box lengths
    if box_lengths['x'] > box_lengths['y']:
        loading = 1
    elif box_lengths['y'] > box_lengths['x']:
        loading = 4
    
    # Start unloading process
    current_stretch = last_stretch
    i = 0
    print("Unloading DN to deformation-free cofiguration")
    
    while not np.isclose(current_stretch, 1):
        i += 1
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        if err:
            breakpoint()
        ## Update the current_stretch
        current_stretch += stretch_increment
        
    print("Unloading completed!!")
    
    return





def get_damaged_G(DN, DN_initial, dim, model, computational_bKuhn, bKuhn, loading):
    """
    Estimate the damaged DN
    
    Inputs:
        DN (clas): current DN object
        DN_initial (clas): current DN object.
        
    Outputs:
        G (float): Current shear modulus
    """
    # Get Nodes
    Nodes, _ = DN.get_nodes_and_bonds()
    
    # Get Bonds with their types
    Bonds, BondCoeffs = DN.get_bonds_and_coeffs(model)
    
    # Get initial box lengths and boundaries
    initial_box, initial_lengths = DN_initial.get_box()
    
    # Get current box length, and calcualte stretches
    _, lengths = DN.get_box()
    stretches = np.array([L / initial_lengths[key] for key, L in lengths.items()])
    
    # Copy current data file and bring it back to F = I
    os.system("cp %s DN_damaged.dat" %DN.data_file)
    pre.bring_back_affinely("DN_damaged.dat", Nodes, Bonds, stretches, initial_box)
    
    # Rewrite main file and run one relaxation step
    Boundary = [str(node) for node in DN.get_boundary()]
    writeMain("main_G.in", "DN_damaged.dat", Boundary, dim, model)
    err = runinc(loading = 1, inc = 0, dl = 0, dim = dim, main_file = "main_G.in");
    DN_damaged = FracNetworkClass("DN_damaged.dat", "test.res", "main_G.in") ## Damaged DN
    
    # Initialise stress and stretch arrays, amd calculate the first element of each
    stretch_array, cauchy_rubbery = [], []
    cauchy_stress = DN_damaged.calculate_stress(dim) * np.power(computational_bKuhn, 3)
    stretch_array.append(1)
    cauchy_rubbery.append(cauchy_stress)
    
    # Create deformation with 4 steps and run
    stretch_increment = 2.5e-2
    i = 0
    stretch = stretch_array[0]
    while not np.isclose(stretch, 1.1):
        ## Run deformatio step
        i += 1
        err = runinc(loading, i + 1, stretch_increment, dim, main_file = "main_G.in")
        stretch = stretch_array[i - 1] + stretch_increment
        stretch_array.append(stretch)
        
        ## Create DN object and calculate current rubbery cauchy stress
        DN_damaged = FracNetworkClass("DN_damaged.dat", "test.res", "main_G.in") ## Damaged DN
        
        cauchy_stress = DN_damaged.calculate_stress(dim) * np.power(computational_bKuhn, 3)
        cauchy_rubbery.append(cauchy_stress)
    
    # Calculate shear modulus
    cauchy_rubbery = FracNetworkClass.render_stress_units(np.array(cauchy_rubbery), bKuhn)
    G = FracNetworkClass.get_shear_modulus(stretch_array, cauchy_rubbery, loading)
    
    
    return G



def run_reduced_inc(data_file, stretch_increment, loading, dim, main_file, max_attempts = 4):
    """
    Run increments of reduced size.
    
    Inputs:
        data_file (str): name of LAMMPS data file
        stretch_increment (float): size of increment.
        loading (int): Type of loading (see runinc for more details).
        dim (int) problem dimension (2 or 3).
        main_file (str): name of LAMMPS input file
        max_attempts (int): allowed number of increment size reduction.
        
    Outputs:
        err (bool)
    """
    # Initialise variables
    current_inc = 0
    attempt = 0
    reduced_inc = stretch_increment * 0.5
    
    # Get number of bonds in the beginning of the time step
    DN = FracNetworkClass(data_file, "test.res", main_file)
    nBonds_beginning = len(DN.get_nodes_and_bonds()[1])
    
    # Run 
    print("Reducing in half the increment size...")
    print("Maximum number of attempts: %d" %max_attempts)
    while current_inc < stretch_increment:
        err = runinc(loading, 0, reduced_inc, dim, main_file)
        if err:
            ## Reduce by half, and try again
            reduced_inc *= 0.5
            attempt += 1
            if attempt > max_attempts:
                print("Maximum number of attempts reached!")
                err = True
            print("Reduced increment did not work. Reducing more and trying again...")
            print("Reduced attempt: %d" %attempt)
        else:
            ## Check if scissions ocurred during the reduced increment
            print("Reduced increment worked!")
            out = relax_unitl_no_scissions(data_file, dim, main_file, nBonds_beginning)
            current_inc += reduced_inc
    
    return err


def relax_unitl_no_scissions(data_file, dim, main_file, nBonds_beginning):
    """
    Relax the network after defomation until scissions are not detected
    anymore.
    
    Inputs:
        data_file (str): name of LAMMPS data file
        dim (int): dimension of the probem (2 or 3)
        main_file (str): name of LAMMPS input file.
        nBonds_beginning (int): number of bonds after deformation.
        
    Outputs:
        None
    """
    
    # Create current DN object
    DN = FracNetworkClass(data_file, "test.res", main_file) ## Netwotk object
    
    ## Check if scissions ocurred, and if yes, relax the network
    nBonds = len(DN.get_nodes_and_bonds()[1])
    scission_detected = nBonds < nBonds_beginning
    
    if scission_detected:
        print("Scissions detected, perfoming relaxation until no more scisison are detected")
        
        while scission_detected:
            ## relax network
            err = runinc(loading = 1, inc = 0, dl = 0, dim = dim, main_file = main_file);
            if not err:
                print("relaxation completed")
            else:
                ## If error was detected, repeat simulation with reduced timestep
                print("relaxation failed, trying with reduced timestep")
                reduce_LAMMPS_timestep(main_file)
                err = runinc(loading = 1, inc = 0, dl = 0, dim = dim, main_file = "small_step.in");
                if err:
                    breakpoint()
                    break
                else:
                    print("relaxation with smaller timestep worked!!")
                
            
            ## update DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
            temp = len(DN.get_nodes_and_bonds()[1])
            scission_detected = temp < nBonds
            print("nChains pre-relaxation: %d" %nBonds)
            print("nChains after-relaxation: %d" %temp)
            nBonds = temp
            if scission_detected:
                print("Doing another relaxation, as scissions were still detected")
    
    return nBonds



def reduce_LAMMPS_timestep(main_file):
    """
    Reduce the time step of LAMMPS integrator.
    
    Inputs:
        main_file (str): name of the original LAMMPS input file.
        
    Outputs:
        None
    """
    
    # Name of temporary main_file
    temp_file = "small_step.in"
    
    # Read all lines from the original main_file
    with open(main_file, "r") as f:
        lines = f.readlines()
        
    # Write the temporari file
    with open(temp_file, "w+") as f:
        for line in lines:
            if not "timestep" in line:
                f.write(line)
            elif "reset_timestep" in line:
                f.write(line)
            else:
                data = line.strip("\n").split("\t")
                deltaT = float(data[1]) / 4
                f.write("timestep\t%g\n" %deltaT)
        
    return



def relax_multi(geometry_file, model, params, dim, temp_file, strengths, 
                            strong_fraction):
    """
    Perform relaxation on as generated network
    
    Inputs: 
        geometry_file (str): path to acces as-generated network
        model (str): chain model to be used.
                    '1': Gaussian chain.
                    '2': FJC (Langevin) chain.
        params (tuple): network parameters.
        dim (int): dimention of the problem
        strengths (tuple): chain strengths
        stron_fraction (float): fraction of strong chains.
    
    Outputs:
        None
    """
    # Unpack tuple based on the model used
    if int(model) == 4:
        bKuhn, NKuhn, nub3 = params
    
    # Extract Nodes, Bonds, Boudnary and BondTypes
    Nodes, Bonds, Boundary, BondTypes = pre.readGeometry(geometry_file)
    rest_lengths = {idx: 0. for idx in Bonds.keys()} ## list of rest lengths (needed to write the data file)
    
    # Calculate computational Kuhh length
    crosslinks = len(Nodes) - len(Boundary)
    computational_bKuhn = np.power(nub3 /(2 * crosslinks), 1/3)
    
    if int(model) == 4:
        computational_params = (computational_bKuhn, NKuhn)
    
    # Assemble 
    BondTypes = pre.sample_weak_and_strong(list(strengths), strong_fraction, NKuhn, Bonds.keys())
    
    # Write data file
    pre.writePositions(temp_file, Nodes, Bonds, Boundary, BondTypes, model, computational_params, rest_lengths)
    
    # Run relaxation
    run_relaxation(dim, temp_file, Boundary, model)
    return



def relax_as_generated_DN(geometry_file, model, params, dim, temp_file):
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
    # Unpack tuple based on the model used
    if int(model) <= 2:
        bKuhn, NKuhn, nub3 = params
    elif int(model) == 4:
        bKuhn, NKuhn, nub3, critical_r_Nb = params
    
    # Extract Nodes, Bonds, Boudnary and BondTypes
    Nodes, Bonds, Boundary, BondTypes = pre.readGeometry(geometry_file)
    rest_lengths = {idx: 0. for idx in Bonds.keys()} ## list of rest lengths (needed to write the data file)
    
    # Calculate computational Kuhh length
    crosslinks = len(Nodes) - len(Boundary)
    computational_bKuhn = np.power(nub3 /(2 * crosslinks), 1/3)
    
    if int(model) <= 2:
        computational_params = (computational_bKuhn, NKuhn)
    elif int(model) == 4:
        computational_params = (computational_bKuhn, NKuhn, critical_r_Nb)
        
    BondTypes = {idx: NKuhn for idx in BondTypes.keys()}
    
    # Write data file
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
            
        # uniaxial in direction 2
        elif loading == 4:
            newline = "fix 1 all deform 1 x volume y delta " + str(-du/2.) + " " + str(du/2)  + " z volume remap x units box\n"
        
        
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
    f.write('minimize\t1e-10 1e-10 1000 10000\n\n')
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