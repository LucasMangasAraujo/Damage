"""
Script running monotonic loading conditions where the network 
is uniform in terms of spring parameters.
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
    rep_folder_names = "rep_DNs", 
    
    
    # Declare chain parameters
    bKuhn, NKuhn = (1, 100)## Kuhn length (nm) and Number of Kuhn segments
    nu = 1e-3 ## chain density in #chains/nm3
    nub3 = nu * pow(bKuhn, 3); ## normalised (via Kuhn length) chain density
    critical_r_Nb = 0.95 ## fraction of the contour length where scission happens
    failure_criterion = 1
    params = (bKuhn, NKuhn, nub3, critical_r_Nb)
    model = '4' ## chain model 
    dim = 3 ## problem dimension
    
    # Declare some parameters for elastic simulations
    elastic_params = bKuhn, NKuhn, nub3
    elastic_model = '2'
    
    # Define load history
    loading = 1
    stretch_increment = 0.1
    
    # Run simulations for the specified number of repeats
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
        results_folder_names = "results", 
        results_file = "data.csv"
        results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3]"
        elastic_file = "data_elastic.csv"
        elastic_comments = "# stretch[0] true[1] nominal[2]"
        
        ## Run now elastic simulation, but withour keeping DN consfigurations
        print("Running elastic simulation for reference")
        out = sim.runsim(geometry_file, elastic_model, elastic_params, dim, loading, np.diff(out[0]), 
                                data_file)
        results_elastic[1] = out
        
    
    # After completion average results
    averaged_results = post.average_fracture_results(results_dict, loading)
    post.write_results(results_folder_names, results_file, results_comments, averaged_results)
    
    # Repeat for elastic simulations
    averaged_results = post.average_elastic_results(results_elastic, loading)
    post.write_results(results_folder_names, elastic_file, elastic_comments, averaged_results)
    
    return


if __name__ == "__main__":
    main()