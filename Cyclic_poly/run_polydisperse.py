"""
Script running cyclic loading conditions where the network 
This code is used to to test what happens when there is a 
Bimodal distribution of chain lengths.

"""

import sys, os 
sys.path.append("..//")
from utils.loading import create_monotonic_load, get_loading_style
import utils.sim_executor as sim
import utils.post_processing as post
from utils.pre_processing import find_peak_stretches, get_long_chain
import numpy as np

def main():
    
    # Declare parameters that are common to all simulations
    nNodes = 25 ## in thousands
    geometries_path = "..//Geometries//Chain_length//Medium//"
    common_name = str(nNodes) + "k_nodes_"
    data_file = "DN.dat" ## name of LAMMPS data file
    
    # Declare if multiple simulations should be done or just one
    nRepeats = 1
    rep_sim_flag = True ## flag indicating that a representative simulation alone should be performed
    if rep_sim_flag:
        import utils.sim_executor_rep as sim_rep 
    
    # Declare comments that will be present in the data file
    results_comments = "# stretch[0], true[1], nominal[2], fraction_broken[3], G[4], r0[5], short[6], long[7]"
    
    # Declare chain parameters
    bKuhn = 1 ## Kuhn length (nm) and Number of Kuhn segments (average)
    critical_r_Nb = 0.7 ## fraction of the contour length where scission happens
    nu = 1e-3 ## chain density in #chains/nm3
    nub3 = nu * pow(bKuhn, 3); ## normalised (via Kuhn length) chain density
    failure_criterion = 1
    model = '4' ## chain model 
    dim = 3 ## problem dimension
    params = (bKuhn, nub3, critical_r_Nb) ## tuple with constant parameters
    
    # Define load history
    loading = 1
    stretch_increment = 0.1
    
    # Define fraction of long chains to be tested
    long_fractions = np.arange(0.1, 1., 0.1)
    N_short = 100 ## fixed number of Kuhn segments in the short chains
    N_average = 150 ## average chain length (same as that of teh monodisperse case)
    
    # Define the peak stresses
    peak_fractions = (0.5, 0.7, 0.9)
    
    for k, phi in enumerate(long_fractions):
        ## print information on the screen
        print(100 * "*")
        print("Running simulations whith the following volume fraction of long chains = %g" %phi)
        print("Size of stretch increment: %g" %stretch_increment)
        print("The short lengths have %g segments" %N_short)
        print("The average chain length is %g " %N_average)
        print("Failure type: %d" %failure_criterion)
        print("bKuhn = %g nm, nub3 = %g" %(bKuhn, nub3))
        
        
        ## Get the length of the long chains
        N_long = get_long_chain(phi, N_short, N_average)
        print("For the current fraction, the long chains should have %g" %N_long)
        chain_lengths = N_short, N_long
        
        ## Assemble folders 
        results_folder_names = "results", "length_effect", str(round(phi, 1)) + "_long"
        
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
            ## Assemble folder to receive rep DNs
            rep_folder_names = "rep_DNs", "length_effect", str(round(phi, 1)) + "_long"
            results_file = "data_rep.csv"
            
            ## Assemble geometry file
            geometry_file = geometries_path + common_name + "1.txt"
            
            ## Based on the rep curve find the set of peak stretches to be tested
            monotonic_path = "..//Monotonic_poly//results//length_effect//" + str(round(phi, 1)) + "_long//" + results_file
            peak_stretches = find_peak_stretches(phi, peak_fractions, monotonic_path)
            
            ## Run representative simulation
            out = sim_rep.runsim_cyclic_bimodal_rep(geometry_file, model, params, dim, loading, 
                                                        stretch_increment, peak_stretches, 
                                                        failure_criterion, data_file, chain_lengths, 
                                                        phi, rep_folder_names)
            results_dict[1] = out
            
            
        
        # After completion average results
        if rep_sim_flag:
            averaged_results = post.average_cyclic_bimodalLength_results(results_dict, loading)
            post.write_results(results_folder_names, results_file, results_comments, averaged_results)
            
        else:
            breakpoint()
        
        
        print(100 * "*")
        print("\n")
        
    
    
    return


if __name__ == "__main__":
    main()