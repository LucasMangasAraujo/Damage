"""
Script defining the NetworkClass.
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.spatial import cKDTree

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
    
    
    def any_path(self, failure_criterion, initial_Nodes):
        """
        Find if there is a path spanning the network.
        
        Inputs:
            failure_criterion (int): failure criterion.
                    1: path spanning the loading direction (assumed direction 1).
                
                
        Outputs
            path_exists (bool): True if at least one path was found.
            
        """
        # Create graph object representing the network
        G = self.create_DN_graph()
        
        # Get ids of boundary nodes
        Boundary = self.get_boundary()
        
        if failure_criterion == 1:
            ## Find to which nodes are on the xx plane
            xx_0 = [node for node in Boundary if np.isclose(initial_Nodes[node][0], 0)] ## plane x = 0
            xx_1 = [node for node in Boundary if np.isclose(initial_Nodes[node][0], 1)] ## plane x = 1
            Boundary_xx = xx_0, xx_1
            
            ## Query the existance of the path
            path_exists = any(nx.has_path(G, source, target) for source in Boundary_xx[0] 
                                for target in Boundary_xx[1])
            if not path_exists: breakpoint()
        
        
        
        return path_exists
    
    
    @staticmethod
    def get_bond_coeffs(data_file):
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
            G.add_edge(n1,n2);
        
        return G

        
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
    
    
    def get_box_lengths(self):
        """
        Get current box bounds.
        
        Inputs:
            
        Outputs:
            box_lengths
        """
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
            
            ## Read y length
            data = f.readline().strip("\n").split(" ")
            box_lengths['y'] = float(data[1]) - float(data[0])
            
            ## Finally z length
            data = f.readline().strip("\n").split(" ")
            box_lengths['z'] = float(data[1]) - float(data[0])
        
        
        return box_lengths
    
    
    
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


class FillerNetworkClass(NetworkClass):
    """
    A class for filled networks inherented from the NetworkClass
    """
    
    def any_filler_missing(self, placed_fillers, initial_lengths, filler_offset):
        """
        Check if there are fillers outside the computational domain.
        NOTE: THIS MIGHT NEED SOME ADAPTATIONS TO IDENTIFY POSSIBLE M
        MISSING CANDIDATES.
        
        Inputs:
            placed_fillers (set): ids of nodes where spheres were placed.
            initial_lengths (dict): initial length of the simulation box.
            
        Outputs:
            is_missing_array (ndarray): boolean array informing whether a 
                                        filler is out of the domain.
        """
        # Query current box lengths
        current_box = self.get_box_lengths()
        stretches = [l / initial_lengths[key] for key, l in current_box.items()]
        
        # Caculate displacement deltas
        delta_displacements = [(stretch - 1) for stretch in stretches]
        box_bounds = [(0 - (du / 2), 1 + (du / 2)) for du in delta_displacements]
        larger_box_bounds = [(-0.1 - (du / 2), 1.1 + (du / 2)) for du in delta_displacements]
        
        # Get DN graph and current position of the nodes
        Nodes, _ = self.get_nodes_and_bonds()
        G = self.create_DN_graph()
        adjency = G.adj
        
        # Loop over the placed fillers set
        is_missing_array = np.zeros(len(placed_fillers), dtype = bool)
        for i, node_idx in enumerate(placed_fillers):
            ## Initialize coordinates array
            coordinates = []
            coordinates.append(Nodes[node_idx])
            
            ## Get filler points and calculate distances
            for n in adjency[node_idx].keys():
                coordinates.append(Nodes[n])
            
            ## Check if any 
            coordinates = np.array(coordinates) ## transform into matrix
            x, y, z = coordinates[:,0], coordinates[:,1], coordinates[:,2]
            mask_x = (x < box_bounds[0][0]) | (x > box_bounds[0][1])
            mask_y = (y < box_bounds[1][0]) | (y > box_bounds[1][1])
            mask_z = (z < box_bounds[2][0]) | (y > box_bounds[2][1])
            flag = np.any(mask_x | mask_y | mask_z)
            
            ## Check if flag was triggered due to rounding errors
            if flag:
                    mask_x = (x < larger_box_bounds[0][0]) | (x > larger_box_bounds[0][1])
                    mask_y = (y < larger_box_bounds[1][0]) | (y > larger_box_bounds[1][1])
                    mask_z = (z < larger_box_bounds[2][0]) | (y > larger_box_bounds[2][1])
                    larger_box_flag = np.any(mask_x | mask_y | mask_z)
                    
                    if larger_box_flag:
                        is_missing_array[i] = flag
                    else:
                        is_missing_array[i] = larger_box_flag
            else:
                is_missing_array[i] = flag
            
        
        
        return is_missing_array
    
    
    def any_filler_overlap(self, placed_fillers, filler_radius, filler_offset):
        """
        Check if there is sphere overlap in the simulation
        
        Inputs:
            placed_fillers (set): ids of nodes where spheres were placed.
            filler_radius (float): radius of the filler
            filler_offset (float): offset of the filler points.
        Outputs:
            
        """
        # Get the filler particles
        coords_of_fillers = self.get_filler_coordinates(placed_fillers)
        
        # Loop over the scan
        is_overlaped_array = np.zeros(len(placed_fillers), dtype = bool)
        overlap_distance = 2 * (filler_radius + filler_offset) ## filler diameter with offset
        for i, node_idx in enumerate(placed_fillers):
            ## Filter ids not being looked
            other_fillers_idx = placed_fillers.difference([node_idx])
            
            ## Get filler centre and the other ones as well
            filler_centre = coords_of_fillers[node_idx]
            existing_centres = np.array(
                                    [coords_of_fillers[idx] for idx in other_fillers_idx]
                                       )
            tree = cKDTree(existing_centres)
            
            ## Query and store results
            distance_query = tree.query_ball_point(filler_centre, overlap_distance)
            is_overlaped_array[i] = len(distance_query) > 0
            
        
        
        return is_overlaped_array
    
    def get_filler_coordinates(self, placed_fillers):
        """
        Get the coordinates of filler particles.
        
        Inputs:
            placed_fillers (set): ids of nodes where spheres were placed.
            
        Outputs:
            coords_of_fillers (dict): idx and coordinates of filler particles
                                      centres.
        """
        # Get current nodes and 
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Only look at 
        coords_of_fillers = {idx: Nodes[idx] for idx in placed_fillers}
        
        return coords_of_fillers
    
    
    def get_distances_filler(self):
        """
        Get distances in the network considering the presence 
        of filler partiplces
        
        Inputs:
            None
        
        Outputs:
            distances (dict): distances of regular chains.
        
        """
        
        # Get Node coordinates and their positions
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Get which bonds are regular
        idx_of_regular_bonds = self.get_regular_bonds()
        filtered_Bonds = {idx: bond for idx, bond in Bonds.items() if idx in idx_of_regular_bonds}
        
        # Scan and store results
        distances = {}
        for idx, bond in filtered_Bonds.items():
            n1, n2 = bond
            vector = Nodes[n1] - Nodes[n2]
            distances[idx] = np.linalg.norm(vector)
        
        return distances
    
    
    def get_regular_bonds(self):
        """
        Get bonds number of regular chains, i.e, not forming filler particles.
        
        Inputs:
            None
            
        Outputs:
            idx_of_regular_bonds (tuple): sequence of indices of regular chains.
        """
        
        # Read file and find the regular bonds
        idx_of_regular_bonds = []
        with open(self.data_file, "r") as f:
            ## Read until spring coefficients are found
            key = f.readline()
            while "Bond Coeffs" not in key:
                key = f.readline()
            
            f.readline() ## empty line
            
            ## Check if hybrid bond style is being used
            data = f.readline().strip("\n").split(" ")
            try:
                float(data[1])
                hybrid_style_flag = False
            except ValueError:
                hybrid_style_flag = True
            
            ## Find the bonds defining the sphere
            while len(data) > 1:
                
                if hybrid_style_flag:
                    if not 'harmonic' in data:
                        idx_of_regular_bonds.append(int(data[0]))
                else:
                    if np.isclose(float(data[-1]), 0):
                        idx_of_regular_bonds.append(int(data[0]))
                    
                data = f.readline().strip("\n").split(" ")
            
        
        return idx_of_regular_bonds
    
    
    
    def get_volume_fraction(nFillers, filler_radius):
        """
        Calculate the volume fraction in the network for a given number of 
        particles and dimensionless filler radius.
        
        Inputs:
            nFillers(int): number of filler networks in the network
            filler_radius (float):
        Outputs:
            
        
        """
        # Calculate the volume fraction
        vol_fraction = 4 * np.pi * np.power(filler_radius, 3) / 3.
        return vol_fraction
    
    
    
    def get_filler_angles_deviations(self, angle_to_pair):
        """
        Get deviation of the filler angles after equilibrium for a 
        given deformation.
        
        Inputs:
            angle_to_pair(dict): angle-to-bond pair map.
            
        Outputs:
            deviations (dict): average and std current-to-initial angle ratio
        
        """
        # Query DN
        Nodes, Bonds = self.get_nodes_and_bonds()
        Angles = self.get_angles_and_triplets()
        
        # Loop over the angle-to-pair map
        temp = defaultdict(list)
        for angle_idx, bonds_idx in angle_to_pair.items():
            ## Get the vectors
            b1, b2 = Bonds[bonds_idx[0]], Bonds[bonds_idx[1]]
            v1 = Nodes[b1[0]] - Nodes[b1[1]]
            v2 = Nodes[b2[0]] - Nodes[b1[1]]
            
            ## Calculate current angle between pairs
            dot_product = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
            theta = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
            
            ## Store in appropriate position the information
            b1, b2 = set(b1), set(b2)
            node = next(iter((b1.intersection(b2))))
            theta0 = Angles[angle_idx][1]
            temp[node].append(np.abs(theta - theta0))
            
        
        # Use the average the results in the dict
        deviations = {key: (np.mean(data), np.std(data)) for key, data in temp.items()}
        
        # Return also maxmum and minimum deviation
        max_key = max(deviations, key=lambda k: deviations[k][0])
        min_key = min(deviations, key=lambda k: deviations[k][0])
        max_avg_deviation = deviations[max_key][0]
        min_avg_deviation = deviations[min_key][0]
        
        return deviations, max_avg_deviation, min_avg_deviation
    
    def get_filler_radii_deviations(self, bond_flags, selected_nodes, filler_radius):
        """
        Get the deviation in filler radii after equilibrium for a given
        deformation.
        
        Inputs:
            bond_flags (dict): flags that inform if bond forms a filler.
            selected_nodes (list): indiced of the nodes acting as the filler centres.
            filler_radius (flaot): expected radius of the filler.
            
        Outputs:
            deviations (list): list of current-to-initial radii ratios.
        """
        # Query DN
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Map filler bonds to their centres
        bond_to_centre_map = defaultdict(list)
        for idx, bond in Bonds.items():
            if all(bond_flags[idx]):
                for node in bond:
                    if node in selected_nodes:
                        bond_to_centre_map[node].append(bond)
            
        
        
        # Calculate deviations
        deviations = [
            np.mean([np.linalg.norm(Nodes[n1] - Nodes[n2]) for (n1, n2) in bond_to_centre_map[node]]) / filler_radius
            for node in selected_nodes
        ]
    
        return deviations
    
    
    
    def get_angles_and_triplets(self):
        """
        Get initial angles and the triplets defining each one of them
        Inputs:
            None
        
        Outputs:
            Angles (dict): triplet of nodes forming the angle and their 
                           initial values. The function will return None
                           if there are not angle restrictions.
        """
        # Check if method is aplicable 
        with open(self.data_file, "r") as f:
            lines = f.readlines()
        NA_flags = ["Angles" in line or "Angle Coeffs" in line for line in lines]
        if not any(NA_flags):
            return None
        
        # Scan file
        with open(self.data_file, "r") as f:
            ## Go through file until angle coefficientes were found
            key = f.readline()
            while "Angle Coeffs" not in key:
                key = f.readline()
            
            f.readline() ##read empty line
            
            ## Read the theta0 values of each angle
            theta0 = {}
            data = f.readline().strip("\n").split(" ")
            while len(data) > 1:
                theta0[int(data[0])] = float(data[2])
                data = f.readline().strip("\n").split(" ")
            
            ## Now scan until the angles section is reached
            key = f.readline()
            while "Angle" not in key:
                key = f.readline()
            
            f.readline() ## empty line 
            
            ## Read list of triplets and to which angle they are linked
            data = f.readline().strip("\n").split(" ")
            triplets = {}
            while len(data) > 1:
                idx = int(data[1])
                triplet = int(data[2]), int(data[3]), int(data[4])
                triplets[idx] = triplet
                data = f.readline().strip("\n").split(" ")
            
        
        # Assamble list of angles
        Angles = {}
        for idx in theta0.keys():
            Angles[idx] = triplets[idx], theta0[idx]
        
        return Angles
        
    
    @staticmethod
    def estimate_nFillers(vol_fraction, filler_radius):
        """
        Estimate the number of fillers particles needed in the network
        for a given volume fraction of fillers and normalize filler 
        radius.
        """
        nFillers = np.ceil(3 * vol_fraction / (4 * np.pi * np.power(filler_radius, 3)))
        return int(nFillers)
    
    @staticmethod
    def estimate_filler_radius(vol_fraction, nFillers):
        """
        Estimate normalised filler radius for a given filler volume
        fraction and number of filler particles in the network
        """
        filler_radius = np.power(3 * vol_fraction / (4 * np.pi * nFillers), 1 / 3)
        return filler_radius
    
    @staticmethod
    def get_filler_par(csv_file):
        """
        Read csv file containing filler parameters
        
        Inputs:
            csv_file (str): full path to acces the file.
            
        Outputs:
            nFillers (int): number of filler particles
            filler_radius (float): radius of the fillers.
            filler_offset (float): filler points offset.
        """
        # Open the data file
        data = np.loadtxt(csv_file, delimiter = ",")
        
        return nFillers, filler_radius, filler_offset