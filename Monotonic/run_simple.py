import sys, os 
sys.path.append("..//")
from utils.network_class import NetworkClass
from utils.loading import create_monotonic_load, get_loading_style
import utils.sim_executor as sim

def main():
    
    # Declare file containing geometry, and name of data file
    geometry_file = '..//Geometries//4000_nodes.txt'
    data_file = "DN.dat"
    
    # Declare chain parameters
    bKuhn, NKuhn = (1, 400)## Kuhn length (nm) and Number of Kuhn segments
    nu = 1e-3 ## chain density in #chains/nm3
    nub3 = nu * pow(bKuhn, 3); ## normalised (via Kuhn length) chain density
    critical_r_Nb = 0.15 ## fraction of the contour length where scission happens
    failure_criterion = 1
    params = (bKuhn, NKuhn, nub3, critical_r_Nb)
    model = '4' ## chain model 
    dim = 3 ## problem dimension
    
    # Define load history
    loading = 1
    stretch_increment = 0.5
    
    stress = sim.runsim_frac(geometry_file, model, params, dim, loading, stretch_increment,
                                failure_criterion, data_file)
    
    
    breakpoint()
    return


if __name__ == "__main__":
    main()