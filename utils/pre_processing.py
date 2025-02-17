"""
Script containing pre-processing functions, such as generation of data files,
reading of input files, etc. See description of each function.
"""
import numpy as np
import random
from pathlib import Path
from .network_class import NetworkClass
from collections import defaultdict
from scipy.spatial import cKDTree


def create_fillers(nFillers, filler_radius, filler_epsilon):
    """
    Create filler particles at random points in the interior of the RVE.
    
    Inputs:
        nFillers (int): number of filler particles.
        filler_radius (float): filler radius 
        filler_epsilon (float) offset of the filler points
        
    Outputs:
        new_Nodes (dict): nodes in the network with the fillers
        new_Bonds (dict): new bonds in the network.
        bond_flags (dict): flags indicating the nature of the bonds (regular, filler, offset)
        new_Angles (dict): angles between intra-filler bonds
        angle_to_pair(dict): map betwen pair of bonds idx and their associated angles
        selected_nodes (list): node number of nodes selected to act as filler.
        
    """
    # Create Network object
    DN = NetworkClass("temp.dat", "test.res", "main.in")
    
    # Get Nodes and Bonds dicts
    Nodes, Bonds = DN.get_nodes_and_bonds()
    nNodes_old = len(Nodes) ## old number of nodes
    nBonds_old = len(Bonds) ## old number of bonds
    
    # Create graph object from the DN
    G = DN.create_DN_graph()
    adjency = G.adj.copy()
    
    # Get boundary nodes, and separte inner from interior nodes
    Boundary_set = set(DN.get_boundary())
    Boundary = [str(node) for node in Boundary_set]
    nodes_idx_set = set([idx for idx in Nodes.keys()])
    inner_nodes_idx = nodes_idx_set.difference(Boundary_set)
    
    # Radomly pick the points that will be replaced by spheres
    selected_nodes = sorted(random.sample(inner_nodes_idx, nFillers))
    
    # Start processs of creatting fillers
    point_replacements = {} ## dict storing the positions of the new nodes, and their global numbering
    point_offsets = {} ## dict associated with the offsets associates with new nodes
    point_angles = {} ## dict storing associated with the angles that will be inserted
    nNodes_added = 0 ## number of added in the final network
    nNodes_new = nNodes_old
    placed_spheres = set() ## set containing the ids of added spheres
    
    for i in range(len(selected_nodes)):
        
        ## Check if sphere overlaps with the ones previously added
        node_idx = check_sphere_overlap(Nodes, placed_spheres, selected_nodes[i], filler_radius, 
                                        filler_epsilon, Boundary_set, selected_nodes)
        placed_spheres.add(node_idx)
        
        ## Call single filler creation function
        filler_points, filler_angles, offset_points = create_filler_points(filler_radius, filler_epsilon,
                                                                            node_idx, adjency, Nodes)
        
        ## Create numbering of the nodes to be added
        filler_points_idx, offset_points_idx =  create_filler_numbering(len(filler_points), len(offset_points), 
                                                                        nNodes_new)
        nNodes_new += 2 * len(filler_points)
        ## Store results and accumulate information 
        point_replacements[node_idx] = filler_points, filler_points_idx
        point_offsets[node_idx] = offset_points, offset_points_idx
        point_angles[node_idx] = filler_angles
        
        ## Delete edges from the selected node
        G.remove_node(node_idx)
    
    # Print on the screen if overlaping spheres were detected
    if not placed_spheres == set(selected_nodes):
        print("Overlaping spheres were detected and replaced by others to avoid overlap")
    else:
        print("No overlapping was detected.")
    
    # Create add new Nodes to teh dict, and create new bonds
    new_Nodes = {idx: coord for idx, coord in Nodes.items()} ## store initial nodes
    new_Angles = {} ## list of angle triplets and the equilibrium angle
    corresponding_bond_pair = {} ## pair of bonds that form the angle
    nAngles = 0 ## number of angles
    added_Bonds = defaultdict(list) ## list of added bonds
    added_flags = defaultdict(list) ## flags associated with the added bonds
    
    
    for node_idx in placed_spheres:
        ## Get list of neighbours from the adjcency
        neighbours = tuple(adjency[node_idx].keys())
        
        ## Add new filler nodes and create bonds with the filler centre
        filler_points = point_replacements[node_idx][0]
        filler_points_idx = point_replacements[node_idx][1]
        for local_idx, global_idx in enumerate(filler_points_idx):
            ## Add filler node to the Nodes dict
            new_Nodes[global_idx] = filler_points[local_idx]
            
            ## Form bond and store it in the new Bonds dict
            bond = [global_idx, node_idx]
            added_Bonds[node_idx].append(bond)
            added_flags[node_idx].append((True, True)) ## second flag informs that bond does not contain offset
        
        ## Repeat process for the offset points
        offset_points = point_offsets[node_idx][0]
        offset_points_idx = point_offsets[node_idx][1]
        for local_idx, global_idx in enumerate(offset_points_idx):
            ## Add offset point to the 
            new_Nodes[global_idx] = filler_points[local_idx]
            
            ## Add filler-to-offset connection
            bond = [global_idx, filler_points_idx[local_idx]]
            added_Bonds[node_idx].append(bond)
            added_flags[node_idx].append((True, False))
            
            ## Add offset-to-node connection
            bond =[global_idx, neighbours[local_idx]]
            added_Bonds[node_idx].append(bond)
            added_flags[node_idx].append((False, False))
        
        ## Form now the tripelts forming angles
        filler_angles = point_angles[node_idx]
        rows, cols = np.triu_indices(filler_angles.shape[0], k = 1) ## off-diagonal upper triangular indices
        
        for i, j in zip(rows, cols):
            nAngles += 1 ## update number of angles
            n1, n2 = filler_points_idx[i], filler_points_idx[j]
            theta0 = filler_angles[i][j]
            triplet = n1, node_idx, n2
            new_Angles[nAngles] = [triplet, theta0]
            corresponding_bond_pair[nAngles] = sorted([n1, node_idx]), sorted([n2, node_idx])
        
    # Create new Bonds dict
    idx = 0 ## bond idx counter
    new_Bonds = {} ## new dict of bonds
    bond_flags = {} ## flags to separate bond types
    angle_to_pair = defaultdict(list)
    
    for bond in G.edges(): ## preserved links in the network
        idx += 1 ## uptade bond counter
        new_Bonds[idx] = sorted(list(bond), reverse = False)
        bond_flags[idx] = (False, False)
    
    for node_idx in added_Bonds.keys(): ## new links representing the fillers
            bonds = added_Bonds[node_idx]
            flags = added_flags[node_idx]
            
            for i, bond in enumerate(bonds):
                idx += 1
                new_Bonds[idx] = bond
                bond_flags[idx] = flags[i]
                matching_keys = [key for key, value in corresponding_bond_pair.items() if sorted(bond) in value]
                for key in matching_keys:
                    angle_to_pair[key].append(idx)
                    
        
    return new_Nodes, new_Bonds, bond_flags, new_Angles, angle_to_pair, Boundary, placed_spheres

def check_sphere_overlap(Nodes, placed_spheres, idx_trial, radius, offset, Boundary, selected_nodes):
    """
    Check if current sphere overlaps with any of the previously placed.
    If that is the case, find replacement for that node.
    
        Nodes (dict): nodes coordinates and their ids
        placed_spheres (set): ids of nodes where spheres have been 
                              succesfully placed
        idx_trial (int): id of nodes where we attempt to place
                         a sphere.
        radius (float): sphere radius.
        offset (float): offeset of sphere points.
        Boundary (set): ids of nodes attached to the RVE boundary.
        selected_nodes (set): ids of nodes initially drawn for sphere
                              placement.
        
    Output:
        node_idx (int): idx of node where the non-overlaping sphere will 
                        be placed.
    """
    
    # First check if placed spheres set is possible
    if len(placed_spheres) == 0:
        return idx_trial
    
    # Calculate distance between the sphere centre and the othen ones 
    sphere_centre = Nodes[idx_trial]
    existing_centres = np.array([Nodes[idx] for idx in placed_spheres])
    #distances = np.sqrt(np.sum((existing_centres - sphere_centre) ** 2, axis=1))
    tree = cKDTree(existing_centres) ## distance tree for fast query
    
    
    # Check if any distances lead to sphere colision. Replace node if needed
    r_plus = radius + offset ## upper bound of the sphere
    distance_query = tree.query_ball_point(sphere_centre, 2 * r_plus)
    any_overlaped = len(distance_query) > 0
    
    if any_overlaped:
        ## Add trial node idx in the list 
        overlaped_idx = set()
        overlaped_idx.add(idx_trial)
        
        ## Do drawing process first doing the difference between Nodes keys
        nodes_idx_set = set([idx for idx in Nodes.keys()])
        available_for_draw = nodes_idx_set.difference(Boundary)
        
        ## Now perform the difference between with respect to the idx of nodes we had
        ## selected before. This ensures that we won't sample already picked nodes.
        available_for_draw = available_for_draw.difference(selected_nodes)
        
        ## Finally sample one node from the available ones.
        drawn_idx = random.sample(available_for_draw, 1)[0]
        
        ## Check whether this node does not yield overlamping spheres.
        repeat_flag = True
        while repeat_flag:
            sphere_centre = Nodes[drawn_idx]
            #distances = np.sqrt(np.sum((existing_centres - sphere_centre) ** 2, axis=1))
            distance_query = tree.query_ball_point(sphere_centre, 2 * r_plus)
            repeat_flag = len(distance_query) > 0
            if repeat_flag:
                
                ## if drawn idx still leads to overlamping repeat sampling
                overlaped_idx.add(drawn_idx)
                available_for_draw = available_for_draw.difference(overlaped_idx)
                try:
                    drawn_idx = random.sample(available_for_draw, 1)[0]
                except ValueError:
                    breakpoint()
                ## checked if newly drawn node makes sense
                sphere_centre = Nodes[drawn_idx]
                #distances = np.sqrt(np.sum((existing_centres - sphere_centre) ** 2, axis=1))
                distance_query = tree.query_ball_point(sphere_centre, 2 * r_plus)
                repeat_flag = len(distance_query) > 0
        
        ## Assign that node to the placed_spheres set
        node_idx = drawn_idx
        
    else:
        
        node_idx = idx_trial
    
    
    return node_idx





def create_filler_numbering(nFiller, nOffset, nNodes_old):
    """
    Create global node numbering of node added to form filler
    
    Inputs:
        nFiller (int): number of filler points
        nOffset (int): number of offset points
        nNodes_old (int): old number of nodes in the network.
    
    Output:
        filler_points_idx (tuple): node numbering of the filler points.
        offset_points_idx (tuple): node numbering of the offset particles
    """
    # Loop over the filler points matrix
    filler_points_idx = tuple(i + nNodes_old + 1 for i in range(nFiller))
    
    # Repeat now for the offsets, considering the filler points
    offset_points_idx = tuple([i + nNodes_old + nFiller + 1 for i in range(nFiller)])
    
    return filler_points_idx, offset_points_idx


def create_filler_points(filler_radius, filler_epsilon, idx_of_central_node, adjency, Nodes):
    """
    Create single filler particle
    
    Inputs:
        filler_radius (float): Radius of the filler particle
        filler_epsilon (float): pertubation of the offset points.
        
    Returns:
        sphere_points (ndarray): points defining the sphere
    """
    # Get nodes connected to central node
    neighbours = tuple(adjency[idx_of_central_node].keys())
    
    # Create filler as a sphere with given radius in the centre of the unit cell
    filler_points, filler_angles = create_sphere_points(Nodes, neighbours, filler_radius, idx_of_central_node);
    
    # Creat offset of associated with the newly added bonds
    offset_points = create_offset_points(filler_points, Nodes[idx_of_central_node], filler_epsilon)
    
    return filler_points, filler_angles, offset_points


def create_sphere_points(Nodes, neighbours, filler_radius, idx_of_central_node):
    """
    Create points on a sphere for the 8chain geometry, and new connections to be placed
    
    Inputs:
        Nodes (dict): Dict containing the coordinates of the nodes in the 8-chain cell
        neighbours (tuple): idx of nodes originally connected to central node
        filler_radius (float): sphere radius.
        idx_of_central_node (int): index of node representing the sphere centre.
        
    Outputs:
        sphere_points (ndarray): coordinates of the points on the sphere
        sphere_angles (npdaary): symmetric and traceless matrix with angles between 
                                 the sphere-to-point vectors.
    """
    
    # Create Nx3 matrix with coordinates of neighbours to central node. (N = functionality)
    neighbour_coords = np.array([Nodes[idx] for idx in neighbours]);
    sphere_centre = Nodes[idx_of_central_node] ## cell centre
    
    # Create vectors
    vectors_to_neighbours = neighbour_coords - sphere_centre ## from sphere centre to cube corners
    unit_vectors_to_neighbours = vectors_to_neighbours / \
                                np.linalg.norm(vectors_to_neighbours, axis = 1)[:, np.newaxis]
    
    # Scale unit vectors by the sphere radius
    sphere_points = (unit_vectors_to_neighbours * filler_radius) + sphere_centre
    
    # Form now angles associated with the sphere
    dot_products = np.dot(unit_vectors_to_neighbours, unit_vectors_to_neighbours.T)
    sphere_angles = np.degrees(np.arccos(np.clip(dot_products, -1, 1)))
    
    return sphere_points, sphere_angles


def create_offset_points(sphere_points, sphere_centre, epsilon):
    """
    Create points representing offsets assocauted with the new created 
    points representing the sphere.
    
    Inputs:
        offset_points (ndarray): Nx3 array with the coordinates of points on the sphere.
        epsilon (float): Magnitude of the pertubation.
        
    Outputs:
        offset_points (nparray): Nx3 array with coordinates of perturbed points.
        
    """
    # Calculate unit vectors pointing in the radial direction
    vector = sphere_points - sphere_centre
    unit_vectors = vector / np.linalg.norm(vector, axis = 1)[:, np.newaxis]
    
    # Created perturbed points
    offset_points = sphere_points + (epsilon * unit_vectors)
    
    return offset_points



def write_data_file(filename, Nodes, Bonds, Angles, Boundary, BondTypes, model, 
                    params, rest_lengths, angle_model , angle_params):

    """
    Writes LAMMPS data file containing the structure and properties of the
    DN.
    
    NOTE: If model != '1', them hybrig bond style will be used. This means
    that bonds need to have their style i the data file.
    
    Inputs:
        filename : name of the file that will be generated
        Nodes : dictionary whose keys are the IDs of the nodes,
                and the values are a list with node coordinates
        Bonds: dictionary whose keys are the bond IDs and the,
                values are a list containing the pair of nodes 
                connected.
        Angles (dict): triplets defining angle and rest angle in 
                       degrees.
        Boundary: list of strings with the IDs of the boundary nodes
        BondTypes: dictionary whose keys are the bond IDs and the,
                values are the chain lengths.
        model : string indicating the type of bond behaviour.
                model = '1': Gaussian
                model = '2': FJC
                model = '3': Breakable extensible FJC
                model = '4': Breakable FJC
                model = '5': Harmonic (Hookean)
                molde = '6': Breakable Gaussian chain
        params: list containing chain parameters other than 
                the chain length.
            
        angle_model: type of angle potential to be used.
                     model = '1': Harmonic
        angle_params: list containing parameters not containing in the 
                      angles dict
    
    The function returns None.
    
    """
    
    # Calculate the number of nodes, bonds, and boundary nodes
    Natoms = len(Nodes);
    Nbonds = len(Bonds);
    Nangles = len(Angles);
    Nboundary = len(Boundary);
    NbondTypes = len(BondTypes);
    
    # Check for polydispersity
    chain_lengths = np.array(list(BondTypes.values()));
    polydispersity_flag = not np.all(chain_lengths == chain_lengths[0])
    if not polydispersity_flag: NbondTypes = 1;
    
    # Open file and write on it
    with open(filename, 'w') as f:
        
        #Header
        f.write('LAMMPS data file for the initial network geometry\n\n');

        #Number of nodes and bonds
        f.write('%d atoms\n' %Natoms);
        f.write('1 atom types\n');
        f.write('%d bonds\n' %Nbonds);
        f.write('%d angles\n' %Nangles);
        f.write('%d bond types\n' %NbondTypes);
        f.write('%d angle types\n\n' %Nangles);

        #Box dimensions 
        f.write('-0.1 1.1 xlo xhi\n');
        f.write('-0.1 1.1 ylo yhi\n');
        f.write('-0.1 1.1 zlo zhi\n\n');

        #Masses
        f.write('Masses\n\n1 1\n\n');

        #Bond coefficients
        f.write('Bond Coeffs\n\n');
        bKuhn = params[0]; ## Kuhn length
        
        if polydispersity_flag:
            for idx, N in BondTypes.items():
                ## Ger rest lengths and calcualte Gaussian stiffness
                r0 = rest_lengths[idx] ## rest length of the bonds
                kappa = (3./2.) * (1 / ( N * pow(bKuhn, 2) )); ## Bond stiffness in the Gaussian regime
                if model in ['1', '5']: ## Gaussian chain (harmonic)
                    if model =='1':
                        f.write('%d %g %g\n'%(idx, kappa, r0))
                    elif model == '6':
                        f.write('%d %g %g %g\n'%(idx, kappa, 0., N * bKuhn)); ## Add the contour length
                    else:
                        rest_length = params[2];
                        f.write('%d %g %g\n'%(idx, kappa, rest_length)); ## zero rest length
                    
                elif model == '2' or model == '4': ## FJC or breakable FJC
                    if np.isclose(r0, 0, atol = 1e-16):
                        f.write('%d langevin %g %g\n' %(idx, bKuhn, N));
                    else:
                        f.write('%d harmonic %g %g\n'%(idx, kappa, r0))
                
                elif model == '3': ## Extensible FJC
                    bKuhn, Eb, critical_eng = tuple(params);
                    f.write('%d %g %g %g %g\n' %(idx, bKuhn, N, Eb, critical_eng));
                
            
        else:
            N = chain_lengths[0];
            
            if model in ['1', '5', '6']: ## Gaussian chain (harmonic)
                
                kappa = (3./2.) * (1 / ( N * pow(bKuhn, 2) )); ## Bond stiffness in the Gaussian regime
                if model == '1':
                    f.write('1 %g %g\n'%(kappa, 0.)); ## zero rest length
                elif model == '6':
                    f.write('1 %g %g %g\n'%(kappa, 0., N * bKuhn)); ## Add the contour length
                else:
                    rest_length = params[2];
                    f.write('1 %g %g\n'%(kappa, rest_length)); ## zero rest length
                
            elif model == '2' or model == '4': ## FJC or breakable FJC
                f.write('1 %g %g\n' %(bKuhn, N));
            
            elif model == '3': ## Extensible FJC
                bKuhn, Eb, critical_eng = tuple(params);
                f.write('1 %g %g %g %g\n' %(bKuhn, N, Eb, critical_eng));
            
        
        f.write('\n\n');
        
        # Write angle coefficients
        f.write('Angle Coeffs\n\n');
        triplet_dict = {idx: triplet for idx, (triplet, _) in Angles.items()} ## extract triplets
        if angle_model == '1':
            kappa_theta = angle_params[0]
            theta0_dict = {idx: angle for idx, (_, angle) in Angles.items()}
            for idx, theta_0 in theta0_dict.items():
                f.write('%d %g %g\n' %(idx, kappa_theta, theta_0))
                
            
        f.write('\n\n');
        
        #Atoms ids and positions
        f.write('Atoms\n\n')
        for idx in Nodes:
            f.write('%d 1 1 %g %g %g\n' %(idx,Nodes[idx][0],Nodes[idx][1],Nodes[idx][2]))
        
        f.write('\n')
        # Bonds IDs and pairs of nodes connected by each bond
        f.write('Bonds\n\n') 

        # Check for polydispersity
        for idx in Bonds:
            if(polydispersity_flag): ## Each bond has its own type
                f.write('%d %d %g %g\n' % (idx,idx,Bonds[idx][0],Bonds[idx][1]) );
            else: ## there is one bond type only
                f.write('%d 1 %g %g\n' %(idx,Bonds[idx][0],Bonds[idx][1]));
            
        
        f.write('\n')
        
        # Ids of the angles and the atom triplet defining them
        f.write('Angles\n\n')
        for idx, triplet in triplet_dict.items():
            f.write('%d %d %d %d %d\n' %(idx, idx, triplet[0], triplet[1], triplet[2]))
            
        f.write('\n')

    return


def assign_bond_types(bond_flags, filler_radius, filler_epsilon, NKuhn, stiffness_ratio):
    """
    Assign spring properties to each bond in the network.
    
    Inputs:
        bond_flags (dict): flags indicating the type of bond
        filler_radius (float): radius of the filler partilcles
        filler_epsilon (float): offset of the filler points
        NKuhn (float): original number of Kuhn segments.
        stiffness_ratio (float): ratio between filler bonds stiffness and that of
                                 refular bonds.
        
    Outputs:
        BondTypes (dict): types of each bond and their chain lengths
        rest_lengths (dict): rest lengths of the bonds. Only relevant for filler bonds
    """
    
    # Create BondTypes containing chains lengths of each type
    BondTypes = {
                idx: NKuhn / stiffness_ratio if bond_flags[idx][0] else NKuhn
                for idx in bond_flags.keys()
                }
    
    # Assemble rest lenghts for the sphere links (TO DO: INCLUDE THIS IN A PRE-PROCESSING FUNCTION)
    rest_lengths = {}
    for idx, bond_type in bond_flags.items():
        ## First check if link is associated with filler
        if bond_type[0]:
            if bond_type[1]:
                rest_lengths[idx] = filler_radius
            else:
                rest_lengths[idx] = filler_epsilon
            
        else:
            rest_lengths[idx] = 0.
    
    
    return BondTypes, rest_lengths

def create_filler_8chain(filler_radius, filler_epsilon):
    """
    Create filler particle in the 8-chain unit cell, and add them to the Nodes 
    and Bonds dict
    
    Inputs:
        filler_radius (float): Radius of the filler particle
        filler_epsilon (float): pertubation of the offset points.
        
    Returns:
        new_Nodes (dict): Dict containing the old coordinates plus new ones from the particle
        new_Bonds (dict): Dict with new connections in the unit cell
        Boundary (list): List with strings containing the ids of nodes at the boundary
    """
    
    # Read file with 8-chain geometry
    out = readGeometry('..//Geometries//8chain.txt')
    Nodes, Bonds = out[0], out[1]
    Boundary = out[2]
    nNodes, nBonds = len(Nodes), len(Bonds)
    idx_of_central_node = 9 ## numbering of central node representing the sphere centre
    
    # Create filler as a sphere with given radius in the centre of the unit cell
    filler_points, filler_bonds, filler_angles = create_sphere_points_8chain(Nodes = Nodes, radius = filler_radius, 
                                                                                idx_of_central_node = 9);
    nFillerPoints, nFillerBonds = len(filler_points), len(filler_bonds)
    
    # Creat pertued filler points
    perturbed_filler_points, perturbed_filler_bonds = perturb_sphere_points_8chain(filler_points, Nodes[9], nNodes, filler_epsilon)
    nPerturbedPoints, nPerturbedBonds = len(perturbed_filler_points), len(perturbed_filler_bonds);
    
    # Add new filler_points to the dict of nodes
    old_node_keys = set(Nodes.keys()) ## store old keys of nodes dict
    new_Nodes = {}
    nNewNodes = nNodes + nFillerPoints + nPerturbedPoints;
    for idx in range(nNewNodes):
        if idx + 1 in old_node_keys:
            new_Nodes[idx + 1] = Nodes[idx + 1]
        elif idx + 1 <= nNodes + nFillerPoints:
            new_Nodes[idx + 1] = filler_points[idx - nNodes, :]
        else:
            new_Nodes[idx + 1] = perturbed_filler_points[idx - nNodes - nFillerPoints, :]
    
    
    # Form new dict of bonds
    aux = list(filler_bonds) + list(perturbed_filler_bonds)
    new_Bonds = {idx + 1: bond for idx, bond in enumerate(aux)}
    nNewBonds = len(new_Bonds)
    
    # Assign bond types to each newly craeted bond
    bond_flags = {};
    old_node_keys.remove(9) ## remove numbering of the central node
    for idx, bond in new_Bonds.items():
        ## Note that two flags are given: non-regular bond, sphere_bond (in sphere)
        if any(n in old_node_keys for n in bond):
            ## Old connections
            bond_flags[idx] = False, False ## only one flag for regular bonds
        else:
            ## Connections between pertubations and the remaining points.
            if idx_of_central_node in bond:
                bond_flags[idx] = True, True
            else:
                bond_flags[idx] = True, False
        
    
    # Create dict containing angles
    new_Angles = {i + 1: triplet_and_angle for i, triplet_and_angle in enumerate(filler_angles)}
    
    
    return new_Nodes, new_Bonds, Boundary, bond_flags, new_Angles

def create_sphere_points_8chain(Nodes, radius, idx_of_central_node):
    """
    Create points on a sphere for the 8chain geometry, and new connections to be placed
    
    Inputs:
        Nodes (dict): Dict containing the coordinates of the nodes in the 8-chain cell
        radius (float): sphere radius.
        idx_of_central_node (int): index of node representing the sphere centre.
        
    Outputs:
        sphere_points (ndarray): coordinates of the points on the sphere
        sphere_bonds (list): connections between the points on the sphere and other points 
                             in the cell.
    """
    
    # Create Nx3 matrix with unit cell vertices coords
    cube_vertices = np.array(tuple(Nodes.values())[:-1]);
    sphere_centre = Nodes[idx_of_central_node] ## cell centre
    nVertices = len(cube_vertices)
    
    # Create vectors
    vectors_to_vertices = cube_vertices - sphere_centre ## from sphere centre to cube corners
    unit_vectors_to_vertices = vectors_to_vertices / \
                                np.linalg.norm(vectors_to_vertices, axis = 1)[:, np.newaxis]
    
    # Scale unit vectors by the sphere radius
    sphere_points = (unit_vectors_to_vertices * radius) + sphere_centre
    
    # Create numbering (1-indexed) of vertices and points on the sphere
    vertices_numbering = tuple(Nodes.keys())
    sphere_points_numbering = np.arange(nVertices + 2, nVertices + len(sphere_points) + 2, 1)
    
    # Create angles of the sphere
    sphere_angles = []
    dot_products = np.dot(unit_vectors_to_vertices, unit_vectors_to_vertices.T)
    angles = np.arccos(np.clip(dot_products, -1, 1))
    
    for i, unit_vector1 in enumerate(unit_vectors_to_vertices):
        for j, unit_vector2 in enumerate(unit_vectors_to_vertices):
            if i == j:
                continue
            ## Generate triplet and check if does not exist already
            triplet = (sphere_points_numbering[i], idx_of_central_node, sphere_points_numbering[j])
            existent_triplets = {tuple(sorted(t[0])) for t in sphere_angles if len(sphere_angles) > 0}
            if tuple(sorted(triplet)) in existent_triplets:
                continue
                
            sphere_angles.append([triplet, np.degrees(angles[i, j])])
        
    
    # Create bonds between the sphere points and its centre
    sphere_bonds = [] ## list of bonds (tuples) initialization
    for i in range(len(sphere_points)):
        bond = sphere_points_numbering[i], idx_of_central_node
        sphere_bonds.append(bond)
    
    # Loop the sphere points to find correspondances between sphere points and vertices
    for i, sphere_point in enumerate(sphere_points):
        ## Get unit vector from centre to point on the sphere
        vector_to_sphere = sphere_point - sphere_centre
        unit_vector_to_sphere = vector_to_sphere / np.linalg.norm(vector_to_sphere)
        
        ## Find vertext that best aligns with the uni vector
        dot_products = np.dot(unit_vectors_to_vertices, unit_vector_to_sphere) ## matrix with dot products
        matching_vertex_idx = np.argmax(dot_products)
        
        ## Create bond between point and corresponding vertices
        bond = sphere_points_numbering[matching_vertex_idx], vertices_numbering[matching_vertex_idx]
        #sphere_bonds.append(bond)
    
    
    return sphere_points, sphere_bonds, sphere_angles

def perturb_sphere_points_8chain(sphere_points, sphere_centre, nNodes, epsilon = 1e-6):
    """
    Create points that pertubed versions of the points on the spheres
    
    Inputs:
        sphere_points (ndarray): Nx3 array with the coordinates of points on the sphere.
        sphere_centre (ndarray:): 3-row array with coordinates of the sphere centre.
        nNodes (int): Original number of nodes in the network.
        epsilon (float, optional): Magnitude of the pertubation.
        
    Outputs:
        perturbed_sphere_points (nparray): Nx3 array with coordinates of perturbed points.
        
    
    """
    # Calculate unit vectors pointing in the radial direction
    vector = sphere_points - sphere_centre
    unit_vectors = vector / np.linalg.norm(vector, axis = 1)[:, np.newaxis]
    
    # Created perturbed points
    perturbed_sphere_points = sphere_points + (epsilon * unit_vectors)
    nPerturbed = len(perturbed_sphere_points);
    
    # Create arrays with global node numbering
    vertices_global_numbering = np.arange(1, nNodes) ## discard sphere centre
    sphere_global_numbering = np.arange(nNodes + 1, nNodes + len(sphere_points) + 1, 1)
    
    # Create bonds sphere-perturbed 
    perturbed_bonds = [];
    for i, node_idx in enumerate(sphere_global_numbering):
        ## Create first sphere-to-pertubation bond
        bond = node_idx + nPerturbed, node_idx
        perturbed_bonds.append(bond)
        
        ## Create now pertubation-to-vertex bond
        bond = node_idx + nPerturbed, vertices_global_numbering[i]
        perturbed_bonds.append(bond)
        
    
    return perturbed_sphere_points, perturbed_bonds

def generate_8chain_geometry(NKuhn, dim = 3):
    """
    Generate geometry file containing unit cell of the 8-chain model
    Inputs:
        NKuhn (float): Number of Kuhn segments in the chain needed for the reading processing
        dim (integer): Optional integer indicating the dimension of the problem (optional)
        
    """
    # Create path to folder that will contain the geomtry file
    parent_dir = Path.cwd().parent
    geom_folder = parent_dir / "Geometries"
    geom_folder.mkdir(exist_ok = True); ## makes sure that the folder exists
    
    # Create node positions
    Nodes = {};
    Nodes[1] = np.array([0.0, 0.0, 0.0]);
    Nodes[2] = np.array([1.0, 0.0, 0.0]);
    Nodes[3] = np.array([1.0, 1.0, 0.0]);
    Nodes[4] = np.array([0.0, 1.0, 0.0]);
    Nodes[5] = np.array([0.0, 0.0, 1.0]);
    Nodes[6] = np.array([1.0, 0.0, 1.0]);
    Nodes[7] = np.array([1.0, 1.0, 1.0]);
    Nodes[8] = np.array([0.0, 1.0, 1.0]);
    Nodes[9] = np.array([0.5, 0.5, 0.5]);
    
    # Create connections dict
    Bonds = {};
    Bonds[1] = (9, 1);
    Bonds[2] = (9, 2);
    Bonds[3] = (9, 3);
    Bonds[4] = (9, 4);
    Bonds[5] = (9, 5);
    Bonds[6] = (9, 6);
    Bonds[7] = (9, 7);
    Bonds[8] = (9, 8);
    
    # Create list of boundary nodes
    Boundary = np.arange(1, 9, 1, dtype = np.int16)
    
    # Create bond types
    Bond_types = {idx: NKuhn for idx in Bonds.keys()}
    
    # Write geometry file
    with open(geom_folder / "8chain.txt", "w+") as f:
        ## Start with nodes coordinates
        f.write("$nodes\n")
        for idx, coord in Nodes.items():
            aux = (idx, coord[0], coord[1],  coord[2]) ## auxiliar tuple for writting
            f.write("%d, %g, %g, %g\n" %aux)
        ## Write connections now
        f.write("$bonds\n")
        for idx, (n1, n2) in Bonds.items():
            aux = idx, n1, n2
            f.write("%d, %d, %d\n" %aux);
        ## Write Boundary
        f.write("$boundary\n")
        for node in Boundary:
            f.write('%d ' %node);
        f.write("\n");
        ## Write chain length distribution
        f.write("$BondTypes\n")
        for idx, chain_length in Bond_types.items():
            aux = idx, chain_length
            f.write("%d, %g\n" %aux);
    
    
    return

def writePositions(filename, Nodes, Bonds, Boundary, BondTypes, model, params, rest_lengths):

    """
    Writes a file containing the architecture of the discrete network 
    and the parameters of each chain in it in a way that LAMMPS can 
    read it and proceed with the energy minimisation process.
    
    
    filename : name of the file that will be generated
    Nodes : dictionary whose keys are the IDs of the nodes,
            and the values are a list with node coordinates
    Bonds: dictionary whose keys are the bond IDs and the,
            values are a list containing the pair of nodes 
            connected.
    Boundary: list of strings with the IDs of the boundary nodes
    BondTypes: dictionary whose keys are the bond IDs and the,
            values are the chain lengths.
    model : string indicating the type of bond behaviour.
            model = '1': Gaussian
            model = '2': FJC
            model = '3': Breakable extensible FJC
            model = '4': Breakable FJC
            model = '5': Harmonic (Hookean)
            molde = '6': Breakable Gaussian chain
    params: list containing chain parameters other than 
            the chain length.
            
    rest_lengths (dict): dict containing the rest length of the bonds.
                         Only relevant for bonds representing fillers.
    
    
    The function returns None.
    
    """
    
    # Calculate the number of nodes, bonds, and boundary nodes
    Natoms = len(Nodes);
    Nbonds = len(Bonds);
    Nboundary = len(Boundary);
    NbondTypes = len(BondTypes);
    
    # Check for polydispersity
    chain_lengths = np.array(list(BondTypes.values()));
    polydispersity_flag = not np.all(chain_lengths == chain_lengths[0])
    if not polydispersity_flag: NbondTypes = 1;
    
    # Open file and write on it
    with open(filename, 'w') as f:
        
        #Header
        f.write('LAMMPS data file for the initial network geometry\n\n');

        #Number of nodes and bonds
        f.write('%d atoms\n' %Natoms);
        f.write('1 atom types\n');
        f.write('%d bonds\n' %Nbonds);
        f.write('%d bond types\n\n' %NbondTypes);

        #Box dimensions
        f.write('-0.1 1.1 xlo xhi\n');
        f.write('-0.1 1.1 ylo yhi\n');
        f.write('-0.1 1.1 zlo zhi\n\n');

        #Masses
        f.write('Masses\n\n1 1\n\n');

        #Bond coefficients
        f.write('Bond Coeffs\n\n');
        bKuhn = params[0]; ## Kuhn length
        
        if polydispersity_flag:
            for idx, N in BondTypes.items():
                
                if model in ['1', '5']: ## Gaussian chain (harmonic)
                    kappa = (3./2.) * (1 / ( N * pow(bKuhn, 2) )); ## Bond stiffness in the Gaussian regime
                    if model =='1':
                        r0 = rest_lengths[idx]
                        if np.isclose(r0, 0):
                            f.write('%d %g %g\n'%(idx, kappa, 0.)); ## zero rest length for regular bonds
                        else:
                            f.write('%d %g %g\n'%(idx, kappa, r0))
                    elif model == '6':
                        f.write('%d %g %g %g\n'%(idx, kappa, 0., N * bKuhn)); ## Add the contour length
                    else:
                        rest_length = params[2];
                        f.write('%d %g %g\n'%(idx, kappa, rest_length)); ## zero rest length
                    
                elif model == '2' or model == '4': ## FJC or breakable FJC
                    f.write('%d %g %g\n' %(idx, bKuhn, N));
                
                elif model == '3': ## Extensible FJC
                    bKuhn, Eb, critical_eng = tuple(params);
                    f.write('%d %g %g %g %g\n' %(idx, bKuhn, N, Eb, critical_eng));
                
            
        else:
            N = chain_lengths[0];
            
            if model in ['1', '5', '6']: ## Gaussian chain (harmonic)
                
                kappa = (3./2.) * (1 / ( N * pow(bKuhn, 2) )); ## Bond stiffness in the Gaussian regime
                if model == '1':
                    f.write('1 %g %g\n'%(kappa, 0.)); ## zero rest length
                elif model == '6':
                    f.write('1 %g %g %g\n'%(kappa, 0., N * bKuhn)); ## Add the contour length
                else:
                    rest_length = params[2];
                    f.write('1 %g %g\n'%(kappa, rest_length)); ## zero rest length
                
            elif model == '2' or model == '4': ## FJC or breakable FJC
                f.write('1 %g %g\n' %(bKuhn, N));
            
            elif model == '3': ## Extensible FJC
                bKuhn, Eb, critical_eng = tuple(params);
                f.write('1 %g %g %g %g\n' %(bKuhn, N, Eb, critical_eng));
            
            
        f.write('\n\n');
        
        #Atoms ids and positions
        f.write('Atoms\n\n')
        for idx in Nodes:
            f.write('%d 1 1 %g %g %g\n' %(idx,Nodes[idx][0],Nodes[idx][1],Nodes[idx][2]))
        
        f.write('\n')
        # Bonds IDs and pairs of nodes connected by each bond
        f.write('Bonds\n\n') 

        # Check for polydispersity
        for idx in Bonds:
            if(polydispersity_flag): ## Each bond has its own type
                f.write('%d %d %g %g\n' % (idx,idx,Bonds[idx][0],Bonds[idx][1]) );
            else: ## there is one bond type only
                f.write('%d 1 %g %g\n' %(idx,Bonds[idx][0],Bonds[idx][1]));

        f.write('\n')

    return


def readBondTypes(input,Nbonds):
    '''
        This fucntion reads the bond types and store them in dictionary 
        called BondTypes when there are more than one bond type
        -------------------------------------------------------------
        Inputs:
            input: File pointer to be read;
            Nbonds: Number of Bonds in the network;
        *************************************************************
            Outputs
            BondTypes: Dic with bonds types
            key: previous line read
    '''
    #================================================================
    ## Variable Declaration
    # Constant Parameters
    R0 = float(0.); R1 = float(1.0);
    # Logicals 
    again = False;
    # Dictionaryes
    BondTypes = {};
    #================================================================
    # Start scannig Process Depending 
    for i in range(1,Nbonds + 1):
        key = input.readline().strip('\n');
        data = key.split();
        BondTypes[i] = float(data[1]);
    
    return BondTypes, key


def readBoundary(input):

    """
    Read the list of boundary nodes from a file with pointer input
    and store them in an array
    """
    key = input.readline().strip(' \n') #read the next line in the input file
    if ("," in key):
        data = key.split(',')
    else:
        data = key.split(' ');

    return data,key

def readBonds(input):

    """
    Read the list of bonds from a file with pointer input
    and store them in a dictionary
    """

    again = True
    bond_dict = {}

    while again is True:
        key = input.readline().strip('\n')  #read the next line in the input file
        if '$' in key:
            again = False
        else:
            data = key.split(',')
            bond_dict[int(data[0])] = [int(data[1]),int(data[2])]

    return bond_dict,key

def readNodes(input):

    """
    Read the list of node from a file with pointer input
    and store them in a dictionary
    """

    #read nodes
    again = True
    node_dict = {}

    while again is True:

        #read the next line in the input file
        key = input.readline().strip('\n')

        if '$' in key:
            again = False
        else:
            data = key.split(',')
            x = float(data[1])
            y = float(data[2])
            z = float(data[3])
            node_dict[int(data[0])] = [float(x),float(y),float(z)]

    return node_dict,key

def readGeometry(filename):

    """ 
    Read network geometry file
    """

    f = open(filename,'r')

    again = True
    key = f.readline().strip('\n')  #read the next line in the input file

    while again is True:

        if 'nodes' in key:
            Nodes,key = readNodes(f)

        elif 'bonds' in key:
            Bonds, key = readBonds(f)

        elif 'boundary' in key:
            Boundary, key = readBoundary(f)
            key = f.readline().strip('\n') # Start Reading Again $
        elif 'BondTypes' in key:
            BondTypes, key = readBondTypes(f,len(Bonds));
        else:
            again = False

    f.close()

    return Nodes,Bonds,Boundary,BondTypes




if __name__ == "__main__":
    main()
