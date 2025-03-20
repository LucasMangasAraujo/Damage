"""
Script containining functions for post processing simulation results
"""

import numpy as np
from scipy.spatial import KDTree
from pathlib import Path
from collections import defaultdict
from .network_class import FracNetworkClass
from scipy import interpolate


def average_cyclic_bimodalLength_results(results_dict, loading):
    """
    Average the results coming from cyclic simulations 
    with determistic chain scission. Note this function is 
    also valid for simulations with one repeat.
    
    Each value of the dict is formed by a tuple containing ndarrays
    containing information in the following order.
        stretch: value of applied stretch.
        cauchy stress: rubbery components of stress.
        nominal stress: nominal components of stress
        fraction_broken_links: self explanatory.
        damaged shear moduli: self explanatory.
    
    Inputs:
        results_dict (dict): results of the simulation of each repeat.
        loading (int): Type of loading used.
    """
    
    # Check if representative simulation was perfomed
    not_one_repeat = len(results_dict.keys()) > 1
    
    # Perform analysis depending on the type of simulation that was done.
    if not_one_repeat:
        ## under dev
        breakpoint()
    else:
        ## No need for averaging.
        stretch_array = results_dict[1][0]
        cauchy_rubbery = results_dict[1][1]
        nominal_rubbery = results_dict[1][2]
        fraction_broken_chains = results_dict[1][3]
        G = results_dict[1][4]
        r0 = results_dict[1][5]
        fraction_broken_short = results_dict[1][6]
        fraction_broken_long = results_dict[1][7]
        
        ## Calculate full stress depending on the loading conditions.
        cauchy_stress, nominal_stress = nonZero_stress(stretch_array, cauchy_rubbery, 
                                                        nominal_rubbery, loading)
        
        ## Assemble output
        averaged_results = (stretch_array, cauchy_stress, nominal_stress, fraction_broken_chains, G, r0,
                            fraction_broken_short, fraction_broken_long
                            )
        
    
    
    return averaged_results




def average_cyclic_results(results_dict, loading):
    """
    Average the results coming from cyclic simulations 
    with determistic chain scission. Note this function is 
    also valid for simulations with one repeat.
    
    Each value of the dict is formed by a tuple containing ndarrays
    containing information in the following order.
        stretch: value of applied stretch.
        cauchy stress: rubbery components of stress.
        nominal stress: nominal components of stress
        fraction_broken_links: self explanatory.
        damaged shear moduli: self explanatory.
    
    Inputs:
        results_dict (dict): results of the simulation of each repeat.
        loading (int): Type of loading used.
    """
    
    # Check if representative simulation was perfomed
    not_one_repeat = len(results_dict.keys()) > 1
    
    # Perform analysis depending on the type of simulation that was done.
    if not_one_repeat:
        ## under dev
        breakpoint()
    else:
        ## No need for averaging.
        stretch_array = results_dict[1][0]
        cauchy_rubbery = results_dict[1][1]
        nominal_rubbery = results_dict[1][2]
        fraction_broken_chains = results_dict[1][3]
        G = results_dict[1][4]
        #r0 = results_dict[1][5]
        
        ## Calculate full stress depending on the loading conditions.
        cauchy_stress, nominal_stress = nonZero_stress(stretch_array, cauchy_rubbery, 
                                                        nominal_rubbery, loading)
        
        ## Assemble output
        averaged_results = stretch_array, cauchy_stress, nominal_stress, fraction_broken_chains, G
        
    
    
    return averaged_results



def average_fracture_results(results_dict, loading):
    """
    Average the results coming from simulations with determistic
    chain scission. Note this function is also valid for simulations
    with one repeat.
    
    Each value of the dict is formed by a tuple containing ndarrays
    containing information in the following order.
        stretch: value of applied stretch.
        cauchy stress: rubbery components of stress.
        nominal stress: nominal components of stress
        fraction_broken_links: self explanatory.
        damaged shear moduli: self explanatory.
    
    Inputs:
        results_dict (dict): results of the simulation of each repeat.
        loading (int): Type of loading used.
    """
    
    # Check if representative simulation was perfomed
    not_one_repeat = len(results_dict.keys()) > 1
    
    # Perform analysis depending on the type of simulation that was done.
    if not_one_repeat:
        ## Find the number of columns (m) - assuming all matrices have the same width
        first_key = list(results_dict.keys())[0]
        num_cols = results_dict[first_key][0].shape[1]
        
        ## Determine the overlapping range of stretch values
        min_stretches = []
        max_stretches = []
        for data, _ in results_dict.values():
            min_stretches.append(np.min(data[:, 0]))  # Assuming stretch is the first column
            max_stretches.append(np.max(data[:, 0]))
        
        ## Use common range where we have enough data points
        common_min_stretch = np.percentile(min_stretches, 75)  # Take the 75th percentile of minimum values
        common_max_stretch = np.percentile(max_stretches, 25)  # Take the 25th percentile of maximum values
        
        ## If the common range is too small, fallback to a more inclusive range
        if common_max_stretch <= common_min_stretch:
            common_min_stretch = np.median(min_stretches)
            common_max_stretch = np.median(max_stretches)
        
        ## Find out the maximum number of points to use for the common grid
        median_density = []
        for data, _ in results_dict.values():
            stretch_range = np.max(data[:, 0]) - np.min(data[:, 0])
            median_density.append(data.shape[0] / stretch_range)
        
        median_pts_per_unit = np.median(median_density)
        target_points = int(median_pts_per_unit * (common_max_stretch - common_min_stretch))
        target_points = max(50, target_points)  # Ensure at least 50 points
        
        ## Create grid that we will interpolate (using the common stretch range)
        target_stretch = np.linspace(common_min_stretch, common_max_stretch, target_points)
        
        ## Initialise some arrays
        interpolated_matrices = []
        Wf_array, PS_array, max_array = [], [], []
        valid_data_counts = np.zeros(target_points)
        
        ## Loop over the the results dict
        for data, PS in results_dict.values():
            ## Create interpolated matrix of the target size
            interp_matrix = np.zeros((target_points, num_cols))
            
            ## Find the valid data within the common range
            valid_idx = (data[:, 0] >= common_min_stretch) & (data[:, 0] <= common_max_stretch)
            valid_data = data[valid_idx]
            
            
            ## Place stretch directly in interpolated matrix, and interpolate the rest
            interp_matrix[:, 0] = target_stretch  # Set the stretch column directly
            for j in range(1, num_cols):  # Start from column 1
                f = interpolate.interp1d(
                        valid_data[:, 0],  # Use stretch as x values
                        valid_data[:, j],  # Use the current column as y values
                        kind='linear',     # Use linear interpolation for robustness
                        bounds_error=False,
                        fill_value=np.nan  # Mark out-of-bounds values as NaN
                    )
                interp_matrix[:, j] = f(target_stretch)
            
            ## Keep track of how many valid data points we have at each position
            valid_data_counts += ~np.isnan(interp_matrix[:, 1])  # Use the first non-stretch column
            
            ## Integrate the stress strain curves
            stretch_array, nominal_stress = data[:,0], data[:,2]
            work_fracture = FracNetworkClass.integrate_stress_strain(nominal_stress, stretch_array)
            Wf_array.append(work_fracture)
            
            ## Append the average  prestretch
            PS_array.append(PS)
            
            ## Append interpolated matrix
            interpolated_matrices.append(interp_matrix)
            
            ## Append failure stretch
            max_array.append(data[:,0].max())
        
        ## Average results
        stacked_matrices = np.stack(interpolated_matrices, axis=0)
        mean_matrix = np.nanmean(stacked_matrices, axis=0)
        std_matrix = np.nanstd(stacked_matrices, axis=0)
        
        ## Removed invalid entrries#
        valid_rows = ~np.isnan(mean_matrix[:, 1])  # Check first non-stretch column
        mean_matrix = mean_matrix[valid_rows]
        std_matrix = std_matrix[valid_rows]
        
        ## Assemble averaged results
        averaged_results = mean_matrix, std_matrix
        
        ## Average work of fracture and pre-stretch
        preStretch = np.mean(PS_array), np.std(PS_array)
        Wf = np.mean(Wf_array), np.std(Wf_array)
        max_stretch = np.mean(max_array), np.std(max_array)
        
        return averaged_results, Wf, preStretch, max_stretch
        
    else:
        ## No need for averaging.
        stretch_array = results_dict[1][0]
        cauchy_rubbery = results_dict[1][1]
        nominal_rubbery = results_dict[1][2]
        fraction_broken_chains = results_dict[1][3]
        G = results_dict[1][4]
        r0 = results_dict[1][5]
        
        ## Calculate full stress depending on the loading conditions.
        cauchy_stress, nominal_stress = nonZero_stress(stretch_array, cauchy_rubbery, 
                                                        nominal_rubbery, loading)
        
        ## Assemble output
        averaged_results = stretch_array, cauchy_stress, nominal_stress, fraction_broken_chains, G, r0
        
        ## Integrate the nominal stress-strain curve
        Wf = FracNetworkClass.integrate_stress_strain(nominal_stress, stretch_array)
        
        ## Get average pre-stretch
        preStretch_distr = results_dict[1][-1]
        preStretch = np.sqrt(np.mean(preStretch_distr**2))
        
        
        return averaged_results, Wf, preStretch



def average_bimodalLength_results(results_dict, loading):
    """
    Average the results coming from simulations with determistic
    chain scission. Note this function is also valid for simulations
    with one repeat. Note as well that this function is dedicated to 
    networks having a bimodal distribution of lengths.
    
    Each value of the dict is formed by a tuple containing ndarrays
    containing information in the following order.
        stretch: value of applied stretch.
        cauchy stress: rubbery components of stress.
        nominal stress: nominal components of stress
        fraction_broken_links: self explanatory.
        damaged shear moduli: self explanatory.
        r0: damaged end-to-end distance.
        fraction_short: fraction of broken short chains.
        fraction_long: fraction of broken long chains.
    
    Inputs:
        results_dict (dict): results of the simulation of each repeat.
        loading (int): Type of loading used.
    """
    
    # Check if representative simulation was perfomed
    not_one_repeat = len(results_dict.keys()) > 1
    
    # Perform analysis depending on the type of simulation that was done.
    if not_one_repeat:
        ## under dev
        breakpoint()
    else:
        ## No need for averaging.
        stretch_array = results_dict[1][0]
        cauchy_rubbery = results_dict[1][1]
        nominal_rubbery = results_dict[1][2]
        fraction_broken_chains = results_dict[1][3]
        G = results_dict[1][4]
        r0 = results_dict[1][5]
        fraction_broken_short = results_dict[1][6]
        fraction_broken_long = results_dict[1][7]
        
        ## Calculate full stress depending on the loading conditions.
        cauchy_stress, nominal_stress = nonZero_stress(stretch_array, cauchy_rubbery, 
                                                        nominal_rubbery, loading)
        
        ## Assemble output
        averaged_results = (stretch_array, cauchy_stress, nominal_stress, fraction_broken_chains, G, r0,
                            fraction_broken_short, fraction_broken_long
                            )
        
        ## Integrate the nominal stress-strain curve
        Wf = FracNetworkClass.integrate_stress_strain(nominal_stress, stretch_array)
        
        ## Get average pre-stretch
        preStretch_distr = results_dict[1][-1]
        preStretch = np.sqrt(np.mean(preStretch_distr**2))
    
    
    return averaged_results, Wf, preStretch



def average_bimodalStrength_results(results_dict, loading):
    """
    Average the results coming from simulations with determistic
    chain scission. Note this function is also valid for simulations
    with one repeat. Note as well that this function is dedicated to 
    networks having a bimodal distribution of strengths.
    
    Each value of the dict is formed by a tuple containing ndarrays
    containing information in the following order.
        stretch: value of applied stretch.
        cauchy stress: rubbery components of stress.
        nominal stress: nominal components of stress
        fraction_broken_links: self explanatory.
        damaged shear moduli: self explanatory.
        r0: damaged end-to-end distance.
        fraction_weak: fraction of broken weak chains.
        fraction_strong: fraction of broken strong chains.
    
    Inputs:
        results_dict (dict): results of the simulation of each repeat.
        loading (int): Type of loading used.
    """
    
    # Check if representative simulation was perfomed
    not_one_repeat = len(results_dict.keys()) > 1
    
    # Perform analysis depending on the type of simulation that was done.
    if not_one_repeat:
        ## under dev
        breakpoint()
    else:
        ## No need for averaging.
        stretch_array = results_dict[1][0]
        cauchy_rubbery = results_dict[1][1]
        nominal_rubbery = results_dict[1][2]
        fraction_broken_chains = results_dict[1][3]
        G = results_dict[1][4]
        r0 = results_dict[1][5]
        fraction_broken_weak = results_dict[1][6]
        fraction_broken_strong = results_dict[1][7]
        
        ## Calculate full stress depending on the loading conditions.
        cauchy_stress, nominal_stress = nonZero_stress(stretch_array, cauchy_rubbery, 
                                                        nominal_rubbery, loading)
        
        ## Assemble output
        averaged_results = (stretch_array, cauchy_stress, nominal_stress, fraction_broken_chains, G, r0,
                            fraction_broken_weak, fraction_broken_strong
                            )
        
        ## Integrate the nominal stress-strain curve
        Wf = FracNetworkClass.integrate_stress_strain(nominal_stress, stretch_array)
        
        ## Get average pre-stretch
        preStretch_distr = results_dict[1][-1]
        preStretch = np.sqrt(np.mean(preStretch_distr**2))
    
    
    return averaged_results, Wf, preStretch




def average_elastic_results(results_dict, loading):
    """
    Average the results coming from elastic simulations. 
    Note this function is also valid for simulations
    with one repeat.
    
    Each value of the dict is formed by a tuple containing ndarrays
    containing information in the following order.
        stretch: value of applied stretch.
        cauchy stress: rubbery components of stress.
        nominal stress: nominal components of stress
    
    Inputs:
        results_dict (dict): results of the simulation of each repeat.
        loading (int): Type of loading used.
    """
    
    # Check if representative simulation was perfomed
    not_one_repeat = len(results_dict.keys()) > 1
    
    # Perform analysis depending on the type of simulation that was done.
    if not_one_repeat:
        ## under dev
        breakpoint()
    else:
        ## No need for averaging.
        stretch_array = results_dict[1][0]
        cauchy_rubbery = results_dict[1][1]
        nominal_rubbery = results_dict[1][2]
        
        ## Calculate full stress depending on the loading conditions.
        cauchy_stress, nominal_stress = nonZero_stress(stretch_array, cauchy_rubbery, 
                                                        nominal_rubbery, loading)
        
        ## Assemble output
        averaged_results = stretch_array, cauchy_stress, nominal_stress
        
    
    
    return averaged_results


def nonZero_stress(stretch_array, cauchy_rubbery, nominal_rubbery, loading):
    """
    Obtain the non-zero stress component based on the BCx.
    
    Inputs:
        stretch_array (ndarray): array with the driving stretches.
        cauchy_rubbery (ndarray): rubbery cauchy stress components
        nominal_rubbery (ndarray): rubbery nominal stress components
        loading (int): type of loading.
                1: uniaxial tension
                2: equi-axial tension
                3: pure shear
    
    Outputs:
        cauchy_stress (ndarray): non-zero cauchy principal stress.
        nominal_stress (ndarray): non-zero nominal principal stress.
    """
    # Array initialisation
    cauchy_stress = np.zeros(len(cauchy_rubbery), )
    nominal_stress = np.zeros(len(nominal_rubbery), )
    
    # Apply boundary conditions
    if loading == 1: 
        ## Obtain Lagrange multiplierand calculate cauchy stress
        Lagrange_multiplier = np.mean(cauchy_rubbery[:, 1:], axis = 1)
        cauchy_stress = cauchy_rubbery[:, 0] - Lagrange_multiplier
        
        ## Calculate the nominal stress
        nominal_stress = nominal_rubbery[:, 0] - (Lagrange_multiplier / stretch_array)
        
    elif loading == 4:
        temp = np.column_stack((cauchy_rubbery[:, 0], cauchy_rubbery[:, 2]), )
        Lagrange_multiplier = np.mean(temp, axis = 1)
        cauchy_stress = cauchy_rubbery[:, 1] - Lagrange_multiplier
        
        nominal_stress = nominal_rubbery[:, 1] - (Lagrange_multiplier / stretch_array)
    
    
    return cauchy_stress, nominal_stress


def write_results(folder_names, results_file, results_comments, results):
    """
    Write results data to results file to folder.
    
    Inputs:
        folder_names (tuple): tuple containing folder names in descending order
        results_file (string): file.extenstion containing the results. .csv files are
                               prefered.
        results_comments(string): comments displayed in the header of the file
        results (tuple or list): iterable containing the results data.
        
    Outputs:
        None
    """
    
    # Check if folder containing results exists
    results_path = Path(*folder_names) ## create folder
    results_path.mkdir(parents = True, exist_ok = True) ## check if folder exists, and create it if not
    
    # Organise data into a ndarray
    if isinstance(results, tuple):
        results_matrix = np.column_stack([np.array(result) for result in results])
    else:
        results_matrix = results
    
    #  Save data to file
    np.savetxt(results_path / results_file, results_matrix, delimiter = ',', 
                    header = results_comments)
    
    return





def rewrite_data_file(bond_coeffs_lines, data_file):
    """
    Rewrite data file when hybrid bond style is used, as LAMMPS 
    do not store them when writting simulation results in the new
    data file.
    """
    
    # Read all the lines in the file
    with open(data_file, "r") as f:
        lines = f.readlines()
        
    # Rewrite data file adding the bond coefficients section
    with open(data_file, "w") as f:
        ## Loop the lines
        for line in lines:
            ## Check if the part where the bond coeffs were supposed to be are there
            if "Angle Coeffs" in line:
                for bond_coeff_line in bond_coeffs_lines:
                    f.write(bond_coeff_line)
                f.write(line)
            else:
                f.write(line)
    
    return


def check_angles(DN, central_node_idx):
    """
    Check average angle change due to minimisation
    
    Inputs:
    
    
    Outputs:
        angle_change (float): average angle change
    """
    
    # Get current positions and connectivity of the network
    Nodes, Bonds = DN.get_nodes_and_bonds()
    Angles = DN.get_angles_and_triplets()
    
    # Perform first filter of the Bonds, and sort them for analysis with angles
    idx_bonds_in_sphere = find_links_in_sphere(Bonds, central_node_idx)
    filtered_Bonds = {idx: tuple(sorted(Bonds[idx])) for idx in idx_bonds_in_sphere}
    
    # Get information idx of links forming the sphere
    bondPair_to_triplet = bondPair_triplet_map(filtered_Bonds, Angles)
    
    # Perform current angle calculations
    angle_deviations = []
    for angle_idx, bonds_idx in bondPair_to_triplet.items():
        ## Obtain first vector
        
        n1, n2 = filtered_Bonds[bonds_idx[0]]
        vector1 = Nodes[n1] - Nodes[n2]
        norm1 = np.linalg.norm(vector1)
        
        ## Obtain second vector
        n1, n2 = filtered_Bonds[bonds_idx[1]]
        vector2 = Nodes[n1] - Nodes[n2]
        norm2 = np.linalg.norm(vector2)
        
        ## Obtain angle
        dot_product = np.dot(vector1 / norm1, vector2 / norm2) ## normalise lengths
        theta = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
        
        ## Calculate deviation
        theta0 = Angles[angle_idx][1]
        angle_deviations.append(np.abs(theta - theta0))
    
    
    return np.mean(angle_deviations)


def bondPair_triplet_map(filtered_Bonds, Angles):
    """
    Obtain correspondance betweem pair of bonds and their corresponding 
    angles
    """
    # Convert angles triplets into bonds
    triplets_to_bonds = {}
    for idx, data in Angles.items():
        triplet = data[0]
        bond1 = tuple(sorted((triplet[0], triplet[1])))
        bond2 = tuple(sorted((triplet[2], triplet[1])))
        triplets_to_bonds[idx] = bond1, bond2
    
    # Now scan the filterd_Bonds dict
    bondPair_to_triplet = defaultdict(list)
    for idx, bond in filtered_Bonds.items():
        for angle_idx, (bond1, bond2) in triplets_to_bonds.items():
            if bond1 == bond or bond2 == bond:
                bondPair_to_triplet[angle_idx].append(idx)
        
    return bondPair_to_triplet





if __name__ == "__main__":
    main()