"""
Script running cyclic loading conditions where the network 
is uniform in terms of spring parameters. This code is used to
to test what happens when there is a Bimodal distribution of chain
strength.

"""

import sys, os 
sys.path.append("..//")
from utils.loading import create_monotonic_load, get_loading_style
import utils.sim_executor as sim
import utils.post_processing as post
import numpy as np

def main():
    
    # Declare file containing geometry, and name of data file
    geometry_file = '..//Geometries//25k_PS1_1.txt'
    data_file = "DN.dat"
    nRepeats = 1
    rep_sim_flag = True ## flag indicating that a representative simulation alone should be performed
    
    
    # Declare chain parameters
    bKuhn, NKuhn = (1, 100)## Kuhn length (nm) and Number of Kuhn segments
    nu = 1e-3 ## chain density in #chains/nm3
    nub3 = nu * pow(bKuhn, 3); ## normalised (via Kuhn length) chain density
    failure_criterion = 1
    model = '4' ## chain model 
    dim = 3 ## problem dimension
    
    
    # Define load history
    loading = 1
    stretch_increment = 0.05
    
    # Define strengths to be tested
    strengths = 0.4, 0.8
    strong_fractions = 0.0, 0.5, 0.7, 1.0
    strong_fractions = 0.5,
    
    ## Define the peak stresses
    peak_stretches = (1.5, 2.0, 2.3),  (2, 4, 8.), (4, 5, 6), (5, 5.5, 6.0)
    peak_stretches = (4, 5, 6.), 
    
    for k, phi in enumerate(strong_fractions):
        ## print information on the screen
        print(100 * "*")
        print("Running simulations whith the following volume fraction of strong chains = %g" %phi)
        print("Size of stretch increment: %g" %stretch_increment)
        print("bKuhn = %g nm, NKuhn = %g, nub3 = %g" %(bKuhn, NKuhn, nub3))
        print("Failure type: %d" %failure_criterion)
        
        ## Assemble parameters
        params = (bKuhn, NKuhn, nub3)
        
        ## Assemble folders 
        results_folder_names = "results", "strength_effect", str(round(phi, 1)) + "_strong"
        results_file = "data.csv"
        
        ## Run simulations for the specified number of repeats
        results_dict = {}
        results_elastic = {}
        if not rep_sim_flag:
            for i in range(0, nRepeats):
                ## Run full simulation
                out = sim.runsim_frac(geometry_file, model, params, dim, loading, stretch_increment,
                                            failure_criterion, data_file)
                ## store simulation results in dict
                results_dict[i + 1] = out
            
            results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3]"
        else:
            ## Assemble folder to receive rep DNs
            rep_folder_names = "rep_DNs", "strength_effect", str(round(phi, 1)) + "_strong"
            
            ## Run representative simulation
            out = sim.runsim_cyclic_multi_rep(geometry_file, model, params, dim, loading, stretch_increment, 
                                                peak_stretches[k], failure_criterion, data_file, strengths, 
                                                phi, rep_folder_names)
            results_dict[1] = out
            
            results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3] G[4]"
            results_file = "data_rep.csv"
            
        
        # After completion average results
        averaged_results = post.average_cyclic_results(results_dict, loading)
        post.write_results(results_folder_names, results_file, results_comments, averaged_results)
        
        
        print(100 * "*")
        print("\n")
        
    
    
    return


if __name__ == "__main__":
    main()