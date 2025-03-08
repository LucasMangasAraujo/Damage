"""
Scritpt containig functions that will run representative simulations
"""

import numpy as np
import os
import utils.pre_processing as pre
import utils.post_processing as post
from .network_class import NetworkClass, FracNetworkClass
from .loading import *
from .sim_executor import *
from pathlib import Path



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
    # for file in rep_path.iterdir():
        # if file.is_file():
            # file.unlink()
    
    # Initialise output array
    stretch_array = []
    cauchy_stress_array = [] 
    nominal_stress_array = []
    fraction_broken_chains = []
    G_array = []
    r0_array = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print(100 * "=")
    relax_as_generated_DN(geometry_file, model, params, dim, data_file)
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, NKuhn, nub3)) ## extract computational params
    
    # Calculate rubbery stress components in the reference configuration
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get pre-stretch of the network, and initial rms distance
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    avg_preStretch = np.sqrt(np.mean(preStretch_distr**2))
    xi = DN_initial.get_interpenetration()
    print("The average pre-stretch is %g and xi = %g" %(avg_preStretch, xi))
    
    # Calculare the shear modulus and initial rms end-to-end distance
    G, rms_r0 = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, initial_Bonds = DN_initial.get_nodes_and_bonds()
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    G_array.append(G)
    r0_array.append(rms_r0)
    breakpoint()
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
            G, rms_r0 = get_damaged_G(DN, DN_initial, dim, model, computational_params[0], bKuhn, loading)
        
        
        ## Append current stress to the stress array
        cauchy_stress_array.append(cauchy_stress)
        nominal_stress_array.append(nominal_stress)
        G_array.append(G)
        r0_array.append(rms_r0)
        
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
                np.array(fraction_broken_chains), np.array(G_array) , np.array(r0_array),
                preStretch_distr
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
        
    
    # Creat folder to receive representative DNs, and remove data from previous simulations
    rep_path = Path(*folder_names) ## create folder
    rep_path.mkdir(parents = True, exist_ok = True)
    for file in rep_path.iterdir():
        if file.is_file():
            file.unlink()
    
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
    for file in rep_path.iterdir():
        if file.is_file():
            file.unlink()
    
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
    
    # Get pre-stretch of the network, and plot average pre-streth and interpenetration
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    avg_preStretch = np.sqrt(np.mean(preStretch_distr**2))
    xi = DN_initial.get_interpenetration()
    print("The average pre-stretch is %g and xi = %g" %(avg_preStretch, xi))
    
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
        
        ## Check if peak stretch is tensile
        if peak < 1:
            print("Peak stretch prescribed is < 1, skipp...")
            continue
        
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
                                    last_stretch, failure_criterion, data_file, poly_flag, 
                                    folder_names):
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
        data_file (int): name of LAMMPS data file.
        folder_names (tuple): string to form path where geometries will be placed
    
    Outputs:
        out (tuple): results of the simulation
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4 and not poly_flag:
        bKuhn, NKuhn, nub3, critical_r_Nb = params
    elif poly_flag: ## network has either different chain lengths of chain strengths
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











if __name__ == "__main__":
    main()