"""
Script defining the NetworkClass.
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline

class NetworkClass:
    """
    A class to assist querying information from discrete networks. 
    This is the parent class, defining methods that are general for
    all types of networks
    """

    def __init__(self, data_file, dump_file, input_file):
        """
        Class constructor
        
        Inputs:
            data_file (str): name of LAMMPS data file
            dump_file (str): name of LAMMPS dump file
            input_file (str): name of LAMMPS inpute file
            
        """
        self.data_file = data_file
        self.dump_file = dump_file
        self.input_file = input_file
    
    @staticmethod
    def get_shear_modulus(stretch, cauchy_rubbery, loading):
        """
        Compute the small strain shear modulus
        
        Inputs:
            stretch (ndarray): streches.
            cauchy_rubbery (ndarray): principal rubbery components with units.
            loading (int): type of loading. see loading.py for more details.
            
        Outputs:
            G (float): shear modulus (with units)
        """
        # Calculate Lagrange multiplier from boundary conditions
        if loading == 1:
            ## Uniaxial tension direction 1
            Lagrange_multiplier = np.mean(cauchy_rubbery[:, 1:], axis = 1)
            cauchy_stress = cauchy_rubbery[:, 0] - Lagrange_multiplier
        elif loading == 4:
            ## Uniaxial tension in direction 2
            temp = np.column_stack((cauchy_rubbery[:, 0], cauchy_rubbery[:, 2]), )
            Lagrange_multiplier = np.mean(temp, axis = 1)
            cauchy_stress = cauchy_rubbery[:, 1] - Lagrange_multiplier
        
        # Interpolate
        spline = UnivariateSpline(stretch, cauchy_stress, s=0)
        dsigma_dlambda = spline.derivative();
        if loading in [1, 4]:
            G = dsigma_dlambda(1.0) / 3
        
        return G
    
    
    
    def get_preStretch_distr(self, model):
        """
        Get the pre-stretch distribution.
        """
        # Read data file
        with open(self.data_file,"r") as f:
            ## Get the Bond types and their coefficients
            key = f.readline()
            while "Bond Coeffs" not in key:
                key = f.readline()
            
            BondCoeffs = {}
            f.readline()
            data = f.readline().strip("\n").split(" ")
            while len(data) > 1:
                if model == '4':
                    BondCoeffs[int(data[0])] = float(data[1]), float(data[2]), float(data[3])
                
                data = f.readline().strip("\n").split(" ")
            
            ## Read node positions
            f.readline() ## read atoms header
            f.readline() ## read skip line
            data = f.readline().strip("\n").split(" ")
            Nodes = {}
            while len(data) > 1:
                Nodes[int(data[0])] = np.array([float(data[3]), float(data[4]), float(data[5])])
                data = f.readline().strip("\n").split(" ")
            
            ## Keep readin until bond section is reached
            key = f.readline()
            while "Bonds" not in key:
                key = f.readline()
            f.readline()
            
            ## Read bonds
            Bonds = {}
            data = f.readline().strip("\n").split(" ")
            while len(data) > 1:
                Bonds[int(data[0])] = int(data[1]), int(data[2]), int(data[3])
                data = f.readline().strip("\n").split(" ")
            
        
        
        # Loop over the Bonds dict
        preStretch_distr = []
        for idx, (bond_type, n1, n2) in Bonds.items():
            v = Nodes[n1] - Nodes[n2]
            r0 = np.linalg.norm(v)
            if model == '4':
                b, N = BondCoeffs[bond_type][0], BondCoeffs[bond_type][1]
            
            lambda0 = r0 / (np.sqrt(N)* b)
            preStretch_distr.append(lambda0)
        
        
        return np.array(preStretch_distr)
    
    
    
    
    @staticmethod
    def get_bond_coeffs_lines(data_file):
        """
        Get lines of the data file containing the bond coefficients
        
        Inputs:
            data_file (str): file name containing the lammps data file.
            
        Outputs:
            bond_coeffs_lines (list): lines of the containing the bond
                                      coefficients.
        """
        bond_coeffs_lines = []
        with open(data_file, "r") as f:
            key = f.readline()
            while "Bond Coeffs" not in key:
                key = f.readline()
            
            bond_coeffs_lines.append(key)
            key = f.readline()
            while "Angle Coeffs" not in key:
                bond_coeffs_lines.append(key)
                key = f.readline()
        
        
        return bond_coeffs_lines
    
    
    def get_computational_params(self, params):
        """
        Get computational params used in the simulation.
        Inputs:
            params (tuple): parameters in with physical units when relevant. 
                            The order is the following:
                                bKuhn: Kuhn length in nm
                                NKuhn: Number of Kuhn segments in the chain.
                                nub3: Normalised chain density in bKuhn3 units.
        
        Outputs:
            computational_params (tuple): parameters in computational units.
        """
        # Extract information DN structure
        Nodes, Bonds = self.get_nodes_and_bonds()
        Boundary = self.get_boundary()
        
        # Unpack input params 
        bKuhn, NKuhn, nub3 = params
        
        # Normalise using the density of crosslinks (subtracting bounary nodes)
        crosslinks = len(Nodes) - len(Boundary)
        upsilonb3 = nub3 / 2
        computational_bKuhn = np.power( upsilonb3 / crosslinks, 1/3)
        
        # Assemple computational_params tuple
        computational_params = (computational_bKuhn, NKuhn)
        
        return computational_params
    
    
    def create_DN_graph(self):
        """
        Turn DN into Graph.
        
        Inputs:
            None
            
        Outputs:
            G (networkx Graph): networkx graph object.
        """
        # Get Network structure
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Create Graph
        G = nx.Graph();
        G.add_nodes_from(Nodes);
        for idx, (n1,n2) in Bonds.items():
            dist = np.linalg.norm(Nodes[n1] - Nodes[n2])
            G.add_edge(n1,n2, weigth = dist);
        
        return G

        
        
    def get_boundary(self):
        
        # Read main file to find idx of the boundary nodes
        with open(self.input_file, "r") as f:
            ## Read file until group of boundary nodes is found
            key = f.readline()
            while 'group' not in key:
                key = f.readline()
            
            ## Loop over split line and store ids of boundary nodes
            data = key.strip('\n').split(" ")
            Boundary = []
            for idx in data:
                try:
                    Boundary.append(int(idx))
                except ValueError:
                    continue
                
            
            
        
        return tuple(Boundary)
    
    def get_stretches(self, initial_distances):
        """
        Get chain streches of regular chains
        
        """
        
        # Get current distances
        distances = self.get_distances()
        
        # Perform calculations
        stretches = {idx: distances[idx] / r0 for idx, r0 in initial_distances.items()}
        
        return stretches
    
    def get_distances(self):
        """
        Get end-to-end distance of the chains at a given configuration.
        
        Inputs: 
            None
            
        Outputs:
            distances (dict): distances of regular chains.
        """
        
        # Get Node coordinates and their positions
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Scan and store results
        distances = {}
        for idx, bond in Bonds.items():
            n1, n2 = bond
            vector = Nodes[n1] - Nodes[n2]
            distances[idx] = np.linalg.norm(vector)
        
        
        return distances

    def get_nodes_and_bonds(self):
        """
        Get the bonds and node coordinates of the network.
        Inputs:
            None
            
        Outputs:
            Nodes (dict): dictionary containing node positions
            Bonds (dict): dictionary containing bonds
        """
        # Get data file containing network 
        filename = self.data_file
        
        # Initialize dict
        Bonds = {}
        Nodes = {}
        
        with open(filename, 'r') as f:
            key = f.readline().strip()

            while 'Atoms' not in key:
                key = f.readline().strip()

            f.readline()  # Skip header

            data = f.readline().split()
            while len(data) > 1:
                idx = int(data[0])
                x = float(data[3])
                y = float(data[4])
                z = float(data[5])
                Nodes[idx] = np.array([x, y, z]) 

                data = f.readline().split()
            
            key = f.readline().strip()

            while 'Bonds' not in key:
                key = f.readline().strip()

            f.readline()  # Skip header
            
            data = f.readline().split()
            while len(data) > 1:
                idx = int(data[0])
                n1 = int(data[2])
                n2 = int(data[3])
                Bonds[idx] = [n1, n2] 

                data = f.readline().split()
        
        return Nodes, Bonds
    
    
    def get_box(self):
        """
        Get current box bounds.
        
        Inputs:
            
        Outputs:
            box_lengths
        """
        box_boundaries = {}
        box_lengths = {}
        # Read file
        with open(self.data_file, "r") as f:
            key = f.readline()
            ## kepp reading until relevant section is reached
            while "xlo" not in key:
                key = f.readline()
            ## Read x length
            data = key.strip("\n").split(" ")
            box_lengths['x'] = float(data[1]) - float(data[0])
            box_boundaries['x'] = float(data[0]), float(data[1])
            
            ## Read y length
            data = f.readline().strip("\n").split(" ")
            box_lengths['y'] = float(data[1]) - float(data[0])
            box_boundaries['y'] = float(data[0]), float(data[1])
            
            ## Finally z length
            data = f.readline().strip("\n").split(" ")
            box_lengths['z'] = float(data[1]) - float(data[0])
            box_boundaries['z'] = float(data[0]), float(data[1])
            
        
        
        return box_boundaries, box_lengths
    
    
    
    def calculate_stress(self, dim):
        """
        Calculate  cauchy stress using virtual work principle.
        
        Inputs:
            dim (int): problem dimension
        
        Outputs:
            S (ndarray): 3x1 array with the principal stresses
        """
        S = np.zeros(3)
        
        with open(self.dump_file, 'r') as f:
            key = f.readline().strip()

            while 'id' not in key:
                key = f.readline().strip()
            
            data = f.readline().split()
            while len(data) > 1:
                if dim == 3:
                    x = float(data[2])
                    y = float(data[3])
                    z = float(data[4])
                    fx = -float(data[5])  # minus sign for reaction force
                    fy = -float(data[6])
                    fz = -float(data[7])

                    S[0] += fx * x
                    S[1] += fy * y
                    S[2] += fz * z

                else:
                    x = float(data[2])
                    y = float(data[3])
                    fx = -float(data[4])  
                    fy = -float(data[5])

                    S[0] += fx * x
                    S[1] += fy * y
                    
                data = f.readline().split()

        return S
    
    
    @staticmethod
    def calculate_nominal_stress(dim, F, cauchy_stress):
        """
        Calculate nominal stress stress using the virtual work principle.
        
        Inputs:
            dim (int): problem dimension
            F (ndarray): deformation gradient
        
        Outputs:
            S (ndarray): 3x1 array with the principal stresses
        """
        # Assemble F^{-T} in the principal space
        F_minusT = np.array([1/stretch for stretch in F])
        
        # Use regular relation to obtain the nominal stress
        nominal = np.zeros(3)
        nominal = cauchy_stress * F_minusT
        
        return nominal
    
    
    
    @staticmethod
    def render_stress_units(stress_array, bKuhn, T = 298):
        """
        Dimensionalise dimenionless stress in kPa
        
        Inputs:
            stress_array (ndarray): rubbery components of stress in b^3/kT units
            bKuhn (float): Kuhn length in nm.
            T (float, default = 298 K): temperature in Kelvin.
            
        Ouputs:
            stress_kPa (ndarray): stress array in kPa.
            
        """
        # Declare Boltzmann constant
        kB = 1.380649e-23
        kT = kB * T ## temperatur in energy units
        
        # Render stress array with J/nm3 units
        stress_J_over_nm3 = stress_array * kT / np.power(bKuhn, 3)
        
        # Convert to kPa
        stress_kPa = stress_J_over_nm3 * 1e24
        return stress_kPa


class FracNetworkClass(NetworkClass):
    """
    A class for networks where detersministic scissions are allowed.
    Inherented from the NetworkClass.
    
    """
    
    def create_DN_graph_strength(self, model):
        """
        Turn DN into Graph, with strand strength as the weights
        
        Inputs:
            None
            
        Outputs:
            G (networkx Graph): networkx graph object.
        """
        # Get Network structure
        Nodes, _ = self.get_nodes_and_bonds()
        Bonds, Coeffs = self.get_bonds_and_coeffs(model)
        
        # Create Graph
        G = nx.Graph();
        G.add_nodes_from(Nodes);
        for idx, (bond_type, n1,n2) in Bonds.items():
            strength = Coeffs[bond_type][-1]
            G.add_edge(n1,n2, weigth = strength);
        
        return G
    
    
    
    def get_bonds_and_coeffs(self, model):
        """
        Get bonds and their coefficients
        """
        
        # Read data file
        with open(self.data_file,"r") as f:
            ## Get the Bond types and their coefficients
            key = f.readline()
            while "Bond Coeffs" not in key:
                key = f.readline()
            
            BondCoeffs = {}
            f.readline()
            data = f.readline().strip("\n").split(" ")
            while len(data) > 1:
                if model == '4':
                    BondCoeffs[int(data[0])] = float(data[1]), float(data[2]), float(data[3])
                
                data = f.readline().strip("\n").split(" ")
            
            
            ## Keep readin until bond section is reached
            key = f.readline()
            while "Bonds" not in key:
                key = f.readline()
            f.readline()
            
            ## Read bonds
            Bonds = {}
            data = f.readline().strip("\n").split(" ")
            while len(data) > 1:
                Bonds[int(data[0])] = int(data[1]), int(data[2]), int(data[3])
                data = f.readline().strip("\n").split(" ")
        
        
        return Bonds, BondCoeffs
    
    @staticmethod
    def integrate_stress_strain(nominal_stress, stretch):
        """
        Integrate the nominal stress-strain curve.
        
        Inputs:
            nominal_stress (ndarray): self-explanatory
            stretch (ndarray): self-explanatory
            
        Outputs:
            W (float): Energy density.
        """
        # Integrate
        W = np.trapz(nominal_stress, stretch)
        
        return W
    
    def remove_ineffective_clusters(self, G):
        """
        Remover clusters that are coiled from the Graph.
        
        Inputs:
            G (networkx graph obj): self-explanatory
            
        Outputs: 
            None
        """
        # Get current positions of the nodes
        Nodes, _ = self.get_nodes_and_bonds()
        
        # Get connected components of the graph
        connected_components = list(nx.connected_components(G))
        
        # Removed the connected components that are coiled
        ineffective_clusters_ids = []
        for i, component in enumerate(connected_components):
            subG = G.subgraph(component)
            sub_edges = list(subG.edges)
            subG_distances = []
            
            for bond in sub_edges:
                n1, n2 = bond
                dist = np.linalg.norm(Nodes[n1] - Nodes[n2])
                subG_distances.append(dist)
            
            ## check if all distances are close to zero
            is_coiled = np.all(np.array(subG_distances) < 1e-6)
            if is_coiled:
                ineffective_clusters_ids.append(i)
        
        # With ids of the ineffective clusters, remove them from the graph
        for i in ineffective_clusters_ids:
            G.remove_nodes_from(connected_components[i])
        
        return
    
    
    def simplified_graph(self, Boundary):
        """
        Simplify the DN graph, removing nodes that do not contribute
        for the "propagation" of information.
        
        Inputs:
            Boundary (set): ids of boundary nodes
            
        Outputs:
            simplified_G (networx graph): simplified graph
            
        """
        # Create graph object
        simplified_G = self.create_DN_graph()
        
        # Find out which nodes have degree zero or one
        flag = True
        while flag:
            ## Find out which nodes have degree one or zero
            nodes_of_interest = set()
            for node in simplified_G.nodes():
                if simplified_G.degree(node) == 0 or simplified_G.degree(node) == 1:
                    nodes_of_interest.add(node)
                    
            ## Remove boundary nodes from the nodes of interest
            selected_nodes = nodes_of_interest.difference(Boundary)
            if not len(selected_nodes) > 0:
                break
                
            ## Remove now node and edges associated with the selected nodes (if not empty)
            for node in selected_nodes:
                simplified_G.remove_node(node)
            
        
        # Remove clusters that are coiled
        self.remove_ineffective_clusters(simplified_G)
        
        
        return simplified_G
    
    
    def any_path(self, failure_criterion, initial_Nodes):
        """
        Find if there is a path spanning the network.
        
        Inputs:
            failure_criterion (int): failure criterion.
                    1: path spanning the loading direction (assumed direction 1).
                
                
        Outputs
            path_exists (bool): True if at least one path was found.
            
        """
        
        # Get ids of boundary nodes
        Boundary = set(self.get_boundary())
        
        # Creat simplified graph object
        G = self.simplified_graph(set(Boundary))
        
        # Remove nodes that are in the boundary but not in the graph anymore
        G_nodes = set(G.nodes())
        filtered_Boundary = Boundary.intersection(G_nodes)
        path_exists = False ## asume path does not exists
        
        # Analyse if failure ocurred
        if failure_criterion == 1:
            ## Find to which nodes are on the xx plane
            xx_0 = [node for node in filtered_Boundary if np.isclose(initial_Nodes[node][0], 0)] ## plane x = 0
            xx_1 = [node for node in filtered_Boundary if np.isclose(initial_Nodes[node][0], 1)] ## plane x = 1
            #Boundary_xx = xx_0, xx_1
            
            ## Get connected components of the current graph
            components = list(nx.connected_components(G))
            for component in components:
                if any(n in component for n in xx_0) and any(n in component for n in xx_1):
                    path_exists = True
                    return path_exists
            
            
            ## Query the existance of the path
            # path_exists = any(nx.has_path(G, source, target) for source in Boundary_xx[0] 
                                # for target in Boundary_xx[1])
            #if not path_exists: breakpoint()
        
        
        
        return path_exists