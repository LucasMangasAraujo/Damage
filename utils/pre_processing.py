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




def find_peak_stretches(stron_fraction, peak_fractions, monotonic_path):
    """
    Return peak stretches for cyclic simulations
    """
    # Acess monotonic data
    data = np.loadtxt(monotonic_path, delimiter = ",")
    stretch, nominal= data[:,0], data[:,2]
    
    # Find the stretch corresponding ot the peak stretch
    idx_peak_stress = np.argmax(nominal)
    peak_stretch = stretch[idx_peak_stress]
    
    # Compute the peak streches
    peak_stretches = [round(alpha * peak_stretch, 1) for alpha in peak_fractions]
    
    
    return tuple(peak_stretches)



def bring_back_affinely(dat_file, Nodes, Bonds, stretches, initial_box):
    """
    Deform affinely the current DN.
    
    Inputs:
        dat_file (str): name of LAMMPS data file.
        Nodes (dict): coordinates of the nodes before deformation
        Bonds (dict): bond types and the nodes connected.
        stretches (ndarray): current principal stretches.
        initial_box (dict): boundaries of the initial box.
        
    Outputs:
        None
    """
    # Calculate stretch increments 
    inc_stretches = np.array([stretch - 1 for stretch in stretches])
    
    # Deform affinelly the crosslinks
    new_Nodes = {}
    for idx, coord in Nodes.items():
        temp = coord.copy()
        temp += inc_stretches / 2
        temp /= stretches
        new_Nodes[idx] = temp
        
    
    # Copy lines of the current DN
    with open(dat_file, "r") as f:
        lines = f.readlines()
        
    # Rewrite the data file
    with open(dat_file, "w") as f:
        ## Re-use some lines
        for line in lines:
            if "xlo" in line:
                f.write("%g %g xlo xhi\n" %tuple(initial_box['x']))
            elif "ylo" in line:
                f.write("%g %g ylo yhi\n" %tuple(initial_box['y']))
            elif "zlo" in line:
                f.write("%g %g zlo zhi\n" %tuple(initial_box['z']))
            elif "Atoms" in line:
                f.write(line)
                f.write("\n")
                break
            else:
                f.write(line)
            
        ## Write the Nodes
        for idx, coord in new_Nodes.items():
            f.write('%d 1 1 %g %g %g\n' %(idx,coord[0],coord[1],coord[2]))
        
        ## Write Bonds
        f.write("\n")
        f.write("Bonds\n\n")
        for idx, (bond_type, n1, n2) in Bonds.items():
            f.write("%d %d %d %d\n" %(idx, bond_type, n1, n2))
    
    return


def sample_weak_and_strong(strengths, strong_fraction, NKuhn, bond_ids):
    """
    Assigns a strength (weak or strong) to each bond in bond_ids while ensuring that
    the fraction of strong bonds matches the specified strong_fraction.
    
    Inputs:
        strengths (list): chain strengths
        stron_fraction (float): fraction of strong chains.
        NKuhn (float): Chain length.
        bond_ids (iterable): ids of the bonds.
    
    Outputs:
        BondTypes (dict): chain length and strength of each bond.
    
    """
    # Ensure strengths are sorted
    strengths = sorted(strengths)
    weak, strong = strengths
    
    # Calculate how many chains should be sampled
    nStrong = int(strong_fraction * len(bond_ids))
    
    # Sample randomly nStrong bonds from the ids of the bonds
    strong_bonds = set(random.sample(bond_ids, nStrong))
    # Assign
    BondTypes = {}
    for idx in bond_ids:
        bond_strength = strong if idx in strong_bonds else weak
        BondTypes[idx] = NKuhn, bond_strength
    
    return BondTypes


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
                    
                elif model == '2': ## FJC or breakable FJC
                    f.write('%d %g %g\n' %(idx, bKuhn, N));
                
                elif model == '4': ## Breakable FJC
                    NKuhn, critical_r_Nb = N ## N in this case is a tuple
                    f.write('%d %g %g %g\n' %(idx, bKuhn, NKuhn, critical_r_Nb));
                
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
                
            elif model == '2': ## FJC or breakable FJC
                f.write('1 %g %g\n' %(bKuhn, N));
                
            elif model == '4':
                ## Check if N is not an array
                try:
                    float(N)
                    critical_r_Nb = params[-1]
                    f.write("1 %g %g %g" %(bKuhn, N, critical_r_Nb))
                except TypeError:
                    NKuhn, critical_r_Nb = N
                    f.write("1 %g %g %g" %(bKuhn, NKuhn, critical_r_Nb))
            
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
