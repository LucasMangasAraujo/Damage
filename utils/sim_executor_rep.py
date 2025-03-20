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
            nBonds = relax_until_no_scissions_rep(data_file, dim, "main.in", nBonds_beginning)
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
    r0_array = []
    fraction_broken_weak = []
    fraction_broken_strong = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print(100 * "=")
    relax_multi(geometry_file, model, params, dim, data_file, strengths, strong_fraction)
    
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
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, _ = DN_initial.get_nodes_and_bonds()
    initial_Bonds, initial_Coeffs = DN_initial.get_bonds_and_coeffs(model)
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Calculare the shear modulus and initial r0
    G, rms_r0 = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    G_array.append(G)
    r0_array.append(rms_r0)
    
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
        fraction_broken_weak.append(0.)
        fraction_broken_strong.append(0.)
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
            nBonds = relax_until_no_scissions_rep(data_file, dim, "main.in", nBonds_beginning, i, rep_path)
            nBonds_beginning = nBonds
        
        ## Update DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
        
        ## Calculate stresses
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
        nominal_stress = NetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
        
        ## Calculate the damaged shear modulus
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
        
        ## Calculate the fraction of broken chains, and query the contribution of each type
        nBonds = len(DN.get_nodes_and_bonds()[1]) 
        fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
        if scission_detected:
            ## Call DN method to find the fraction of wach type broken
            weak_fraction, strong_fraction = DN.compute_fractions_weak_strong(initial_Bonds, initial_Coeffs, strengths)
            fraction_broken_weak.append(weak_fraction)
            fraction_broken_strong.append(strong_fraction)
        else:
            fraction_broken_weak.append(fraction_broken_weak[-1])
            fraction_broken_strong.append(fraction_broken_strong[-1])
        
        ## Assess if failure occurred
        path_exists = DN.any_path(failure_criterion, initial_Nodes)
        if path_exists:
            print("Failure not detected. Move to next increment")
            print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
            print("%g of the broken chains were weak" %(100 * fraction_broken_weak[-1]))
            print("%g of the broken chains were strong" %(100 * fraction_broken_strong[-1]))
            
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
                np.array(fraction_broken_chains), np.array(G_array), np.array(r0_array), 
                np.array(fraction_broken_weak), np.array(fraction_broken_strong),
                preStretch_distr
           )
    
    return out





def runsim_frac_bimodal_rep(geometry_file, model, params, dim, loading, stretch_increment, 
                             failure_criterion, data_file, chain_lengths, long_fraction, 
                             folder_names):
    """
    Run full simulation for DN where on-and-off scissions are allowed to 
    happen. We run the simulation until failure is detected. This function
    was desgined for cases where there is a bimodal distribution of chain
    lengths.
    
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
        bKuhn, nub3, critical_r_Nb = params 
        
    
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
    r0_array = []
    fraction_broken_short = []
    fraction_broken_long = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print("The fraction of long chains is" %long_fraction)
    print("The short and long chains have %g and %g segments, respectively" %chain_lengths)
    relax_bimodal(geometry_file, model, params, dim, data_file, chain_lengths, long_fraction)
    print(100 * "=")
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, chain_lengths[0], nub3)) ## extract computational params
    computational_bKuhn = computational_params[0]
    
    # Calculate rubbery stress components in the reference configuration
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_bKuhn, 3)
    nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get pre-stretch of the network, and initial rms distance
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    avg_preStretch = np.sqrt(np.mean(preStretch_distr**2))
    xi = DN_initial.get_interpenetration()
    print("The average pre-stretch is %g and xi = %g" %(avg_preStretch, xi))
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, _ = DN_initial.get_nodes_and_bonds()
    initial_Bonds, initial_Coeffs = DN_initial.get_bonds_and_coeffs(model)
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Calculare the shear modulus and initial r0
    G, rms_r0 = get_damaged_G(DN_initial, DN_initial, dim, model, computational_bKuhn, bKuhn, loading)
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    G_array.append(G)
    r0_array.append(rms_r0)
    
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
        fraction_broken_short.append(0.)
        fraction_broken_long.append(0.)
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
            nBonds = relax_until_no_scissions_rep(data_file, dim, "main.in", nBonds_beginning)
            nBonds_beginning = nBonds
        
        ## Update DN object
        DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
        
        ## Calculate stresses
        cauchy_stress = DN.calculate_stress(dim) * np.power(computational_bKuhn, 3)
        nominal_stress = NetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
        
        ## Calculate the damaged shear modulus
        if scission_detected:
            G, rms_r0 = get_damaged_G(DN, DN_initial, dim, model, computational_bKuhn, bKuhn, loading)
            
        
        ## Append current stress to the stress array
        cauchy_stress_array.append(cauchy_stress)
        nominal_stress_array.append(nominal_stress)
        G_array.append(G)
        r0_array.append(rms_r0)
        
        ## Print currrent step data
        print("The fraction of long chains is %g" %long_fraction)
        print("The short and long chains have %g and %g segments, respectively" %chain_lengths)
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        print("S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        print("P_11 = %g, P_22 = %g, P_33 = %g" %tuple(nominal_stress))
        print("G = %g kPa" %G)
        
        ## Move geometry to representative folder
        current_data_file = "Step_" + str(i) + ".dat"
        os.system("cp %s %s" %(data_file, current_data_file))
        os.system("mv %s %s" %(current_data_file, rep_path))
        
        ## Calculate the fraction of broken chains, and query the contribution of each type
        nBonds = len(DN.get_nodes_and_bonds()[1]) 
        fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
        if scission_detected:
            ## Call DN method to find the fraction of wach type broken
            short_frac, long_frac = DN.compute_fractions_short_long(initial_Bonds, initial_Coeffs, chain_lengths)
            fraction_broken_short.append(short_frac)
            fraction_broken_long.append(long_frac)
        else:
            fraction_broken_short.append(fraction_broken_short[-1])
            fraction_broken_long.append(fraction_broken_long[-1])
        
        ## Assess if failure occurred
        path_exists = DN.any_path(failure_criterion, initial_Nodes)
        if path_exists:
            print("Failure not detected. Move to next increment")
            print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
            print("%g of the broken chains were short" %(100 * fraction_broken_short[-1]))
            print("%g of the broken chains were long" %(100 * fraction_broken_long[-1]))
            
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
                np.array(fraction_broken_chains), np.array(G_array), np.array(r0_array), 
                np.array(fraction_broken_short), np.array(fraction_broken_long),
                preStretch_distr
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
    G, rms_r0 = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
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
                nBonds = relax_until_no_scissions_rep(data_file, dim, "main.in", nBonds_beginning)
                nBonds_beginning = nBonds
                
            ## Update DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
            
            ## Calculate stresses
            cauchy_stress = DN.calculate_stress(dim) * np.power(computational_params[0], 3)
            nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
            
            ## Calculate the damaged shear modulus if scissions were detected
            if scission_detected:
                G, rms_r0 = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
            
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
    G, rms_r0 = get_damaged_G(DN_initial, DN_initial, dim, model, computational_params[0], bKuhn, loading)
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
                nBonds = relax_until_no_scissions_rep(data_file, dim, "main.in", nBonds_beginning)
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



def runsim_cyclic_bimodal_rep(geometry_file, model, params, dim, loading, stretch_increment, 
                                peak_stretches, failure_criterion, data_file, chain_lengths, 
                                long_fraction, folder_names):
    """
    Run cyclic simulations for the prescribed peak stetches. This function
    was desgined for cases where there is a bimodal distribution of chain
    lengths in the network.
    
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
        chain_lengths (tuple): short and long chains strengths.
        long_fraction (float): fraction of long chains in the network.
        folder_names (tuple): string to form path where geometries will be placed
    
    Outputs:
        out (tuple): results of the simulations
        
    """
    
    # Unpack parameters tuple, which mighht vary depending of the chain model
    if int(model) == 4:
        bKuhn, nub3, critical_r_Nb = params 
    
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
    r0_array = []
    fraction_broken_short = []
    fraction_broken_long = []
    i = 0 ## increment counter
    
    # Relax as generated network
    print("Starting simulation for network in file %s" %geometry_file)
    print("The fraction of long chains is" %long_fraction)
    print("The short and long chains have %g and %g segments, respectively" %chain_lengths)
    relax_bimodal(geometry_file, model, params, dim, data_file, chain_lengths, long_fraction)
    print(100 * "=")
    
    # Create initial DN object
    os.system("cp %s DN_ref.dat" %data_file)
    DN_initial = FracNetworkClass("DN_ref.dat", "test.res", "main.in") ## reference configuration
    computational_params = DN_initial.get_computational_params((bKuhn, chain_lengths[0], nub3)) ## extract computational params
    computational_bKuhn = computational_params[0]
    _, ref_lengths = DN_initial.get_box()
    
    # Get initial number of nodes and initial coordinates
    initial_Nodes, _ = DN_initial.get_nodes_and_bonds()
    initial_Bonds, initial_Coeffs = DN_initial.get_bonds_and_coeffs(model)
    initial_nBonds = len(initial_Bonds)
    nBonds_beginning = initial_nBonds
    
    # Calculate rubbery stress components in the reference configuration
    cauchy_stress = DN_initial.calculate_stress(dim) * np.power(computational_params[0], 3)
    nominal_stress = NetworkClass.calculate_nominal_stress(dim, np.ones_like(cauchy_stress), cauchy_stress)
    
    # Get pre-stretch of the network, and initial rms distance
    preStretch_distr = DN_initial.get_preStretch_distr(model)
    avg_preStretch = np.sqrt(np.mean(preStretch_distr**2))
    xi = DN_initial.get_interpenetration()
    print("The average pre-stretch is %g and xi = %g" %(avg_preStretch, xi))
    
    
    # Calculare the shear modulus and initial r0
    G, rms_r0 = get_damaged_G(DN_initial, DN_initial, dim, model, computational_bKuhn, bKuhn, loading)
    
    # Append initial information
    stretch_array.append(1.)
    cauchy_stress_array.append(cauchy_stress)
    nominal_stress_array.append(nominal_stress)
    G_array.append(G)
    r0_array.append(rms_r0)
    
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
        fraction_broken_short.append(0.)
        fraction_broken_long.append(0.)
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
                
            ## Initialise DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object
            
            ## Calculate current deformatio gradient
            stretch = DN.obtain_stretch(loading, ref_lengths)
            F = deformation_gradient(loading, stretch)
            stretch_array.append(stretch)
            
            ## Check if scissions ocurred, and if yes, relax the network
            nBonds = len(DN.get_nodes_and_bonds()[1])
            scission_detected = nBonds < nBonds_beginning
            if scission_detected:
                nBonds = relax_until_no_scissions_rep(data_file, dim, "main.in", nBonds_beginning)
                nBonds_beginning = nBonds
                
            ## Update DN object
            DN = FracNetworkClass(data_file, "test.res","main.in") ## Netwotk object 
            
            ## Calculate stresses
            cauchy_stress = DN.calculate_stress(dim) * np.power(computational_bKuhn, 3)
            nominal_stress = FracNetworkClass.calculate_nominal_stress(dim, F, cauchy_stress)
            
            ## Calculate the damaged shear modulus if scissions were detected
            if scission_detected:
               G, rms_r0 = get_damaged_G(DN, DN_initial, dim, model, computational_bKuhn, bKuhn, loading)
            
            ## Append current stress to the stress array
            cauchy_stress_array.append(cauchy_stress)
            nominal_stress_array.append(nominal_stress)
            G_array.append(G)
            r0_array.append(rms_r0)
            
            ## Print currrent step data
            print("The fraction of long chains is %g" %long_fraction)
            print("The short and long chains have %g and %g segments, respectively" %chain_lengths)
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
            
            ## Calculate the fraction of broken chains, and query the contribution of each type
            nBonds = len(DN.get_nodes_and_bonds()[1]) 
            fraction_broken_chains.append((initial_nBonds - nBonds) / initial_nBonds)
            if scission_detected:
                ## Call DN method to find the fraction of wach type broken
                short_frac, long_frac = DN.compute_fractions_short_long(initial_Bonds, initial_Coeffs, chain_lengths)
                fraction_broken_short.append(short_frac)
                fraction_broken_long.append(long_frac)
            else:
                fraction_broken_short.append(fraction_broken_short[-1])
                fraction_broken_long.append(fraction_broken_long[-1])
            
            ## Assess if failure
            path_exists = DN.any_path(failure_criterion, initial_Nodes)
            if path_exists:
                print("Failure not detected. Move to next increment")
                print("%g percent of chains are broken" %(100 * fraction_broken_chains[-1]))
                print("%g of the broken chains were short" %(100 * fraction_broken_short[-1]))
                print("%g of the broken chains were long" %(100 * fraction_broken_long[-1]))
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
                np.array(fraction_broken_chains), np.array(G_array), np.array(r0_array), 
                np.array(fraction_broken_short), np.array(fraction_broken_long)
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
                nBonds = relax_until_no_scissions_rep(data_file, dim, "main.in", nBonds_beginning)
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




def relax_until_no_scissions_rep(data_file, dim, main_file, nBonds_beginning, step, rep_path):
    """
    Relax the network after defomation until scissions are not detected
    anymore.
    
    NOTE: This code is adapted from the original function contained in 
    sim_executor. The difference is that we output the geometry files
    after each relaxatio step.
    
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
    counter = 0
    
    if scission_detected:
        print("Scissions detected, perfoming relaxation until no more scisison are detected")
        
        ## Move the goemetry at the beginning of teh fix def relaxation step
        current_data_file = "Step_" + str(step) + "_inter" +str(counter) +".dat"
        os.system("cp %s %s" %(data_file, current_data_file))
        os.system("mv %s %s" %(current_data_file, rep_path))
        
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
                ## Move teh current geometry file to folder
                counter += 1
                current_data_file = "Step_" + str(step) + "_inter" +str(counter) +".dat"
                os.system("cp %s %s" %(data_file, current_data_file))
                os.system("mv %s %s" %(current_data_file, rep_path))
            
    
    ## If no scissions occured in the relaxation step delete initial file
    if counter == 0:
        file_path = rep_path / current_data_file
        os.remove(file_path)
    
    return nBonds






if __name__ == "__main__":
    main()