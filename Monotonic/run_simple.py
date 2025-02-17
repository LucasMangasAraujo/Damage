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
    critical_r_Nb = 0.95 ## fraction of the contour length where scission happens
    failure_type = 1
    params = (bKuhn, NKuhn, nub3, critical_r_Nb, failure_type)
    model = '2' ## chain model 
    dim = 3 ## problem dimension
    
    # Define load history
    loading = 1
    max_stretch = 5
    increments = 10
    stretch_array, stretch_increment = create_monotonic_load(max_stretch, increments)
    
    stress = sim.runsim(geometry_file, model, params, dim, loading, stretch_array, 
                        stretch_increment)
    
    
    breakpoint()
    return


if __name__ == "__main__":
    main()