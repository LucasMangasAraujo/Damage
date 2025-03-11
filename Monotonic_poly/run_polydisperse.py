"""
Script running monotonic loading conditions where the network 
has a Bimodal distribution of chain lengths. The chain strength
is considered the same for all chains.
"""

import sys, os 
sys.path.append("..//")
from utils.loading import create_monotonic_load, get_loading_style
import utils.sim_executor as sim
import utils.post_processing as post
from utils.pre_processing import get_long_chain
import numpy as np

def main():
    
    # Declare file containing geometry, and name of data file
    data_file = "DN.dat"
    nRepeats = 1
    rep_sim_flag = True ## flag indicating that a representative simulation alone should be performed
    if rep_sim_flag:
        import utils.sim_executor_rep as sim_rep
        geometry_file = '..//Geometries//Chain_length//Medium//25k_nodes_1.txt'
    
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
    long_fractions = np.arange(0.25, 1., 0.25)
    long_fractions = np.arange(0.1, 1., 0.1)
    N_short = 100 ## fixed number of Kuhn segments in the short chains
    N_average = 150 ## average chain length (same as that of teh monodisperse case)
    
    # Initialise arrays, and start loop process
    Wf_array = [] ## array to store array to store the work of fracture
    PS_array = [] ## array to store pre-stretches
    
    for phi in long_fractions:
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
        results_folder_names = "results", "length_effect", str(round(phi, 2)) + "_long"
        
        ## Run simulations for the specified number of repeats
        results_dict = {}
        results_elastic = {}
        if not rep_sim_flag:
            results_file = "data.csv"
            for i in range(0, nRepeats):
                ## Run full simulation
                out = sim.runsim_frac(geometry_file, model, params, dim, loading, stretch_increment,
                                            failure_criterion, data_file)
                ## store simulation results in dict
                results_dict[i + 1] = out
            
            results_comments = "# stretch[0] true[1] nominal[2] fraction_broken[3]"
        else:
            ## Assemble folder to receive data files and name of results file
            rep_folder_names = "rep_DNs", "length_effect", str(round(phi, 2)) + "_long"
            results_comments = "# stretch[0], true[1], nominal[2], fraction_broken[3], G[4], r0[5], short[6], long[7]"
            results_file = "data_rep.csv"
            
            ## Run representative simulation
            out = sim_rep.runsim_frac_bimodal_rep(geometry_file, model, params, dim, loading, 
                                                    stretch_increment, failure_criterion, 
                                                    data_file, chain_lengths, phi, 
                                                    rep_folder_names)
            results_dict[1] = out
            
            
        
        # After completion average results
        averaged_results, Wf, preStretch = post.average_bimodalLength_results(results_dict, loading)
        post.write_results(results_folder_names, results_file, results_comments, averaged_results)
        
        # Average the work of fracture results
        if isinstance(Wf, float):
            PS_array.append(preStretch)
            Wf_array.append(Wf)
        else:
            breakpoint()
        
        print(100 * "*")
        print("\n")
        
    
    # Save results for the work of fracture
    results_folder_names = "results", "length_effect"
    results_file = "Wf.csv"
    if isinstance(Wf, float):
        results_comments = "phi[0], lambda0[1], Wf[2]"
    averaged_results = long_fractions, PS_array, Wf_array
    post.write_results(results_folder_names, results_file, results_comments, averaged_results)
    
    return


if __name__ == "__main__":
    main()