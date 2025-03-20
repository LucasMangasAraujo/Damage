"""
Script running monotonic loading conditions where networks with similar
pre-stretches, but different chain lengths and densities
"""

import sys, os 
sys.path.append("..//")
from utils.loading import create_monotonic_load, get_loading_style
import utils.sim_executor as sim
import utils.post_processing as post
import numpy as np

def main():
    # Declare parameters that are common to all simulations
    nNodes = 25 ## in thousands
    geometries_path = "..//Geometries//Chain_length//"
    common_name = str(nNodes) + "k_nodes_"
    data_file = "DN.dat" ## name of LAMMPS data file
    
    # Declare if multiple repeat should be done, or representative simulations
    nRepeats = 1
    rep_sim_flag = False ## flag indicating that a representative simulation alone should be performed
    if rep_sim_flag:
        import utils.sim_executor_rep as sim_rep
    
    results_comments = "# stretch[0], true[1], nominal[2], fraction_broken[3], G[4], r_0[5]"
    
    # Define load type and increment size
    loading = 1
    stretch_increment = 0.1
    
    # Declare chain parameters
    bKuhn = 1 ##Kuhn length (nm)
    critical_r_Nb = 0.7 ## fraction of the contour length where scission happens
    failure_criterion = 1
    model = '4' ## chain model 
    dim = 3 ## problem dimension
    nub3 = 1e-3 ## normalised chain densities
    
    # Declare densities and chain lengths tested
    chain_lengths = np.array([100, 150, 200])
    length_folders = "Short", "Medium", "Long"
    nRepeats = 5 ## number of pre-stretches used
    
    ## Initliase the array for storing information
    Wf_array = [] ## array to store the densities
    PS_array = [] ## array to store the pre-stretches
    max_array = [] ## array to store the failure stretches
    
    
    for i, NKuhn in enumerate(chain_lengths):
        ## Assemble parameters tuple, and path to geometries
        params = (bKuhn, NKuhn, nub3, critical_r_Nb)
        full_geom_path = geometries_path + length_folders[i] + "//"
        
        ## print information on the screen
        print(100 * "*")
        print("Running simulations whith the following normalised density = %g" %nub3)
        print("Size of stretch increment: %g" %stretch_increment)
        print("bKuhn = %g nm, NKuhn = %g, critical_r_Nb = %g" %(bKuhn, NKuhn, critical_r_Nb))
        print("Failure type: %d" %failure_criterion)
        
        ## Assemble results folder 
        results_folder_names = "results", "length_effect", length_folders[i]
        
        ## Run simulations for the specified number of repeats
        results_dict = {}
        
        if not rep_sim_flag:
            for repeat in range(0, nRepeats):
                ## Assemble geometry file
                geometry_file = full_geom_path + common_name + str(repeat + 1) +".txt"
                
                ## Run full simulation
                out = sim.runsim_frac(geometry_file, model, params, dim, loading, stretch_increment,
                                            failure_criterion, data_file)
                
                ## store simulation results in dict
                results_dict[repeat + 1] = out
                
            
        else:
            ## Assemble file and other folders or files
            geometry_file = full_geom_path + common_name + "1.txt"
            rep_folder_names = "rep_DNs", "length_effect", length_folders[i]
            
            ## Run representative simulation
            out = sim_rep.runsim_frac_rep(geometry_file, model, params, dim, loading, 
                                            stretch_increment,failure_criterion, data_file,
                                            rep_folder_names)
            results_dict[1] = out
            
            
        
        ## Write output results depending wheter a rep simulation was selected
        if rep_sim_flag:
            ## After completion average results
            averaged_results, Wf, preStretch = post.average_fracture_results(results_dict, loading)
            
            results_file = "data_rep.csv"
            post.write_results(results_folder_names, results_file, results_comments, averaged_results)
        else:
            ## Average the results
            averaged_results, Wf, preStretch, max_stretch = post.average_fracture_results(results_dict, loading)
            
            ## Save file with teh averaged data
            results_file = "data_avg.csv"
            post.write_results(results_folder_names, results_file, results_comments, averaged_results[0])
            
            ## Now for the deviations
            results_file = "data_std.csv"
            post.write_results(results_folder_names, results_file, results_comments, averaged_results[1])
            
        
        ## Average the work of fracture results
        PS_array.append(preStretch)
        Wf_array.append(Wf)
        max_array.append(max_stretch)
        
        
        print(100 * "*")
        print("\n")
        
        
    # Save results for the work of fracture
    results_folder_names = "results", "length_effect",
    results_comments = "N[0], lambda0[1], Wf[2], lambda_max[3]"
    if isinstance(Wf, float):
        results_file = "Wf_rep.csv"
        averaged_results = chain_lengths, PS_array, Wf_array
        post.write_results(results_folder_names, results_file, results_comments, averaged_results)
    else:
        ## Get average and std
        avg_PS = [PS[0] for PS in PS_array]
        std_PS = [PS[1] for PS in PS_array]
        avg_Wf = [Wf[0] for Wf in Wf_array]
        std_Wf = [Wf[1] for Wf in Wf_array]
        avg_max = [max_stretch[0] for max_stretch in max_array]
        std_max = [max_stretch[1] for max_stretch in max_array]
        
        ## Save the average data
        results_file = "Wf_avg.csv"
        averaged_results = chain_lengths, avg_PS, avg_Wf, avg_max
        post.write_results(results_folder_names, results_file, results_comments, averaged_results)
        
        ## Save the deviations
        results_file = "Wf_std.csv"
        averaged_results = chain_lengths, std_PS, std_Wf, std_max
        post.write_results(results_folder_names, results_file, results_comments, averaged_results)
    
    return


if __name__ == "__main__":
    main()