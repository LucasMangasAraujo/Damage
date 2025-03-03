"""
Control script responsible to run anisotropy tests for monodisperse
networks with Bimodal chain strength distribution.
"""

import sys, os 
sys.path.append("..//")
from utils.loading import create_monotonic_load, get_loading_style
import utils.sim_executor as sim
import utils.post_processing as post
import numpy as np




def main():
    
    # Declare parameters that are common to all simulations
    geometry_file = '..//Geometries//25k_PS1_1.txt'
    data_file = "DN.dat" ## name of LAMMPS data file
    
    # Declare if multiple repeat should be done, or representative simulations
    nRepeats = 1
    rep_sim_flag = True ## flag indicating that a representative simulation alone should be performed
    if rep_sim_flag:
        results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3]"
    else:
        results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3] G[4]"
    
    # Declare chain parameters
    bKuhn, NKuhn = (1, 100)## Kuhn length (nm) and Number of Kuhn segments
    nu = 1e-3 ## chain density in #chains/nm3
    nub3 = nu * pow(bKuhn, 3); ## normalised (via Kuhn length) chain density
    failure_criterion = 1
    model = '4' ## chain model 
    dim = 3 ## problem dimension
    
    # Define load type and increment size
    loading_xx = 1
    loading_yy = 4
    stretch_increment = 0.05
    
    
    # Define strengths to be tested
    strengths = 0.4, 0.8
    strong_fractions = 0.0, 0.6, 0.7, 1.0
    strong_fractions = 0.5,
    
    ## Define the peak stresses
    peak_stretches = ((2.5, ), (1.9, )), ((4.5, ), (4.0, )), ((6., ), (4., )), ((6., ), (5.5, ))
    
    for k, phi in enumerate(strong_fractions):
        ## print information on the screen
        print(100 * "*")
        print("Running simulations whith the following volume fraction of strong chains = %g" %phi)
        print("Size of stretch increment: %g" %stretch_increment)
        print("bKuhn = %g nm, NKuhn = %g, nub3 = %g" %(bKuhn, NKuhn, nub3))
        print("Failure type: %d" %failure_criterion)
        
        ## Assemble folders 
        results_folder_names = "results", "strength_effect", str(round(phi, 1)) + "_strong"
        
        ## Run simulations for the specified number of repeats
        results_dict_xx = {}
        results_dict_yy = {}
        
        ## Assemble parameters
        params = (bKuhn, NKuhn, nub3)
        
        if not rep_sim_flag:
            for repeat in range(0, nRepeats):
                ## Run full simulation
                out = sim.runsim_frac(geometry_file, model, params, dim, loading, stretch_increment,
                                            failure_criterion, data_file)
                ## store simulation results in dict
                results_dict[repeat + 1] = out
            
        else:
            
            ## Run representative simulation in the x direction
            print("Loading in direction 1")
            rep_folder_names_xx = "rep_DNs", "strength_effect", str(round(phi, 1)) + "_strong", "xx"
            results_file_xx = "data_rep_xx.csv"
            out_xx = sim.runsim_cyclic_multi_rep(geometry_file, model, params, dim, loading_xx, stretch_increment, 
                                                    peak_stretches[k][0], failure_criterion, data_file, strengths, 
                                                    phi, rep_folder_names_xx)
            
            ## Run representative simulation in the y direction
            rep_folder_names_yy = "rep_DNs", "strength_effect", str(round(phi, 1)) + "_strong", "yy"
            results_file_yy = "data_rep_yy.csv"
            out_yy = sim.runsim_cyclic_anisotropy_rep(model, params, dim, loading_yy, stretch_increment,
                                                        peak_stretches[k][1], out_xx[0][-1], failure_criterion, 
                                                        data_file, rep_folder_names_yy)
            
            ## Run ciclic simulation in the 
            results_dict_xx[1] = out_xx
            results_dict_yy[1] = out_yy
            
            
        
        # After completion average results for x the direction
        averaged_results = post.average_cyclic_results(results_dict_xx, loading_xx)
        post.write_results(results_folder_names, results_file_xx, results_comments, averaged_results)
        
        # Do the same for the y direction
        averaged_results = post.average_cyclic_results(results_dict_yy, loading_yy)
        post.write_results(results_folder_names, results_file_yy, results_comments, averaged_results)
        
        print(100 * "*")
        print("\n")
            
        
    
    return




if __name__ == "__main__":
    main()
