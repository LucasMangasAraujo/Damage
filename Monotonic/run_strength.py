"""
Script running monotonic loading conditions where the network 
is uniform in terms of spring parameters. This code is used to
run a set of simulations where the chain strength is changed
"""

import sys, os 
sys.path.append("..//")
from utils.network_class import NetworkClass
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
    
    # Declare some parameters for elastic simulations
    elastic_params = bKuhn, NKuhn, nub3
    elastic_model = '2'
    
    # Define load history
    loading = 1
    stretch_increment = 0.1
    
    # Define strengths to be testes
    strengths = 0.2, 0.4, 0.6, 0.8, 0.5
    strengths = 0.95,
    
    
    for critical_r_Nb in strengths:
    
        ## print information on the screen
        print(100 * "*")
        print("Running simulations where the chains fail at r /(Nb) = %g" %critical_r_Nb)
        print("Size of stretch increment: %g" %stretch_increment)
        print("bKuhn = %g nm, NKuhn = %g, nub3 = %g" %(bKuhn, NKuhn, nub3))
        print("Failure type: %d" %failure_criterion)
        
        ## Assemble parameters
        params = (bKuhn, NKuhn, nub3, critical_r_Nb)
        
        ## Assemble folders 
        rep_folder_names = "rep_DNs", "strength_effect", str(critical_r_Nb) + "_rNb"
        results_folder_names = "results", "strength_effect", str(critical_r_Nb) + "_rNb"
        results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3]"
        
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
            
        else:
            ## Run representative simulation
            out = sim.runsim_frac_rep(geometry_file, model, params, dim, loading, stretch_increment,
                                            failure_criterion, data_file, rep_folder_names)
            results_dict[1] = out
            
            ## Declare destination folde of the results 
            results_file = "data.csv"
            
        
        # After completion average results
        averaged_results, Wf, preStretch = post.average_fracture_results(results_dict, loading)
        post.write_results(results_folder_names, results_file, results_comments, averaged_results)
        
        print(100 * "*")
        print("\n")


        print(100 * "*")
        print("\n")

        print(100 * "*")
        print("\n")
        print(100 * "*")
        print("\n")
    
    return


if __name__ == "__main__":
    main()