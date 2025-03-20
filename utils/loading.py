"""
Script containing functions responsible to generate loading informatio and variables
"""
import numpy as np



def check_loop_stability(previous_peak, current_peak):
    """
    Check hysteresis loop is stable
    """
    # Check if current peak stress is equal to the previous one
    updated_peak_stress = 0.
    if np.isclose(previous_peak, current_peak, atol = 1e-6):
        ## Loop is stable
        print("Loop has stabelised")
        is_stable = True
    elif current_peak < previous_peak:
        print("Loop has stabelised")
        print("The current peak is smaller than the previous one")
        updated_peak_stress = current_peak
        is_stable = False
    else:
        print("Loop has stabelised")
        print("The current peak is greater than the previous one")
        updated_peak_stress = current_peak
        is_stable = False
    
    # Print previois and current peaks on the screen
    print("Previous rubbery peak: %g" %previous_peak)
    print("Current rubbery peak: %g" %current_peak)
    
    return is_stable, updated_peak_stress

def reverse_load(peak, stretch, previous_stretch_increment):
    """
    Check if peak was reached
    """
    # Check if peak was reached
    
    if np.isclose(peak, stretch):
        ## Invert loading direction and set peak flag to true
        current_stretch_increment = -previous_stretch_increment
        is_peak = True
        
        ## Print informatio on the screen
        print("Peak stretch reached, reversing load")
        
    elif np.isclose(1, stretch):
        ## Invert loading direction and set peak flag to False
        current_stretch_increment = -previous_stretch_increment
        is_peak = False
        
        ## Print information on the screen
        print("Material is unloaded.")
        
    else:
        ## Nothing to do
        current_stretch_increment = previous_stretch_increment
        is_peak = False
        
        ## Print information on the screen
        print("Peak not reached nor material is unloaded.")
    
    return current_stretch_increment, is_peak


def get_loading_style(loading):
    """
    Get loading style based on the loading input.
    
    Inputs:
        loading (int): integer identifying the loading type.
                       1: uniaxial tension
                       2: biaxial tension (equi-biaxial)
                       3: pure shear
                       
    Outputs:
        loading_style (str): name of the loading type.
        
    """
    
    if loading == 1:
        loading_style = 'uniaxial tension'
    elif loading == 2:
        loading_style = 'bi(equi)-axial tension'
    elif loading == 3:
        loading_style = 'pure shear'
    
    return loading_style




def create_monotonic_load(max_stretch, increments):
    """
    Create array with monotonic loading history and corresponding increment sizes.
    
    Inputs:
        max_stretch (float): target stretch.
        increments (int): number of increments.
        
    Outputs:
        stretch_array (nparray): array with stretch history.
        stretch_increment (float): stretch increment.
    """
    # Create loading history
    stretch_array = np.linspace(1, max_stretch, increments + 1, endpoint = True)
    
    # Calculate (average) stretch
    stretch_increment = np.mean(np.diff(stretch_array))
    
    return stretch_array, stretch_increment

def deformation_gradient(loading, stretch):
    """
    Compute deformation gradient in the principal space.
    
    Inputs:
        loading (int): identifier of the type of loading (see below):
            1: uniaxial tension
            2: biaxial tension TO DO
            3: pure shear TO DO
            4: uniaxial tension in direction 2 (yy)
            
        stretch (float): scalar charactherising the load "magnitude".
        
    Outputs:
        F (ndarray): 3-row array with the principal stretches
        
    """
    # Initialise
    F = np.zeros(3, )
    
    # Allocate
    if loading == 1:
        F[0] = stretch
        F[1], F[2] = 1 / np.sqrt(stretch), 1 / np.sqrt(stretch)
    elif loading == 4:
        F[1] = stretch
        F[0], F[2] = 1 / np.sqrt(stretch), 1 / np.sqrt(stretch)
    
    return F