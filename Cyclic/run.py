"""
Control script responsible to run cyclic loads in monodisperse
networks and with uniform chain strengths.
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
    geometries_path = "..//Geometries"
    common_name = str(nNodes) + "k_PS"
    data_file = "DN.dat" ## name of LAMMPS data file
    
    # Declare if multiple repeat should be done, or representative simulations
    nRepeats = 1
    rep_sim_flag = True ## flag indicating that a representative simulation alone should be performed
    if rep_sim_flag:
        results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3]"
    else:
        results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3]"
    
    # Define load type and increment size
    loading = 1
    stretch_increment = 0.1
    peak_stretches = (2.5, 3, 4), (3, 4, 4.5), (4, 5, 5.3)
    
    # Declare chain parameters
    bKuhn, NKuhn = (1, 100)## Kuhn length (nm) and Number of Kuhn segments
    critical_r_Nb = 0.7 ## fraction of the contour length where scission happens
    failure_criterion = 1
    model = '4' ## chain model 
    dim = 3 ## problem dimension
    
    # Declare densities tested
    nub3_array = 5e-4, 1e-3, 3e-3 ## normalised chain densities
    nPreStretches = 1 ## number of pre-stretches used
    
    
    for i, nub3 in enumerate(nub3_array):
        ## Assemble parameters tuple, and path to geometries
        params = (bKuhn, NKuhn, nub3, critical_r_Nb)
        full_geom_path = geometries_path + "//nu" + str(i + 1) + "//"
        
        ## Initliase the array for storing information
        Wf_array = [] ## array to store 
        PS_array = []
        
        ## Loop over the pre-stretches
        for j in range(nPreStretches):
            
            ## print information on the screen
            print(100 * "*")
            print("Running simulations whith the following normalised density = %g" %nub3)
            print("The peak stresses are")
            print("Running pre-stretch number %d" %(j + 1))
            print("Size of stretch increment: %g" %stretch_increment)
            print("bKuhn = %g nm, NKuhn = %g, critical_r_Nb = %g" %(bKuhn, NKuhn, critical_r_Nb))
            print("Failure type: %d" %failure_criterion)
            
            ## Assemble folders 
            rep_folder_names = "rep_DNs", "density_effect", "nu" + str(i + 1), "PS" + str( j+ 1)
            results_folder_names = "results", "density_effect", "nu" + str(i + 1), "PS" + str(j + 1)
            results_file = "data_rep.csv"
            
            ## Run simulations for the specified number of repeats
            results_dict = {}
            results_elastic = {}
            
            if not rep_sim_flag:
                for repeat in range(0, nRepeats):
                    ## Run full simulation
                    out = sim.runsim_frac(geometry_file, model, params, dim, loading, stretch_increment,
                                                failure_criterion, data_file)
                    ## store simulation results in dict
                    results_dict[repeat + 1] = out
                
            else:
                ## Assemble file
                geometry_file = full_geom_path + common_name + str(j + 1) + "_1.txt"
                ## Run representative simulation
                out = sim.runsim_cyclic_rep(geometry_file, model, params, dim, loading, 
                                                stretch_increment, peak_stretches[i], 
                                                failure_criterion, data_file, 
                                                rep_folder_names)
                results_dict[1] = out
                
                
            
            # After completion average results
            averaged_results = post.average_cyclic_results(results_dict, loading)
            post.write_results(results_folder_names, results_file, results_comments, averaged_results)
            
            print(100 * "*")
            print("\n")
            
        
    
    return






if __name__ == "__main__":
    main()





