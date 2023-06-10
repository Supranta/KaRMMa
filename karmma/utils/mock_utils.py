import numpy as np
import healpy as hp
import h5py as h5
from scipy.interpolate import interp1d
from scipy.stats import norm, poisson

def check_file_exists(filename):
    import os.path
    if os.path.isfile(filename):
        print('DATAFILE ALREADY EXISTS!')
        return True
    return False

def save_datafile(datafile, g1_obs, g2_obs, k_arr, N, mask):
    file_exists = check_file_exists(datafile)
    overwrite = False
    if(file_exists):      
        overwrite_response = input("WE WILL NEED TO OVERWRITE THE EXISTING DATAFILE. ARE YOU SURE YOU WANT TO PROCEED? (y/n)")
        overwrite_response = overwrite_response.lower()
        assert overwrite_response in ['y', 'n'], "Invalid response"
        overwrite = (overwrite_response == 'y')
        if not overwrite:
            print("Not overwriting the existing file")
            return
    if not file_exists or overwrite:
        if(file_exists):
            print("OVERWRITING FILE!")
        with h5.File(datafile, 'w') as f:
            f['g1_obs'] = g1_obs
            f['g2_obs'] = g2_obs
            f['kappa']  = k_arr
            f['N']      = N    
            f['mask']   = mask               

def get_mock_data(nside, nbins, data):    
    nbar, sigma_e, g1, g2, mask = data
    N_bar = nbar * hp.nside2pixarea(nside, degrees=True) * 60**2

    N = []
    for i in range(nbins):
        N_i = poisson(N_bar[i]).rvs(hp.nside2npix(nside))
        N.append(N_i * mask)
    N = np.array(N)

    sigma = sigma_e / np.sqrt(N + 1e-25)
    g1_obs = g1 + np.random.standard_normal(sigma.shape) * sigma
    g2_obs = g2 + np.random.standard_normal(sigma.shape) * sigma
    g1_obs = g1_obs * mask
    g2_obs = g2_obs * mask        
    return g1_obs, g2_obs, N      

def get_nz_interpolators(nz_data, nbins):
    nz_interpolators = []
    for i in range(nbins):
        interp_nz = interp1d(nz_data[i,:,0], nz_data[i,:,1])
        nz_interpolators.append(interp_nz)
    return nz_interpolators

def get_hsc_z_slice_weights(nz_interpolators, nbins):
    hsc_z_slices = np.array([0.0506,0.1023,0.1553,0.2097,0.2657,
                             0.3233,0.3827,0.4442,0.5078,0.5739,
                             0.6425,0.7140,0.7885,0.8664,0.9479,
                             1.0334,1.1233,1.2179,1.3176,1.4230,
                             1.5345,1.6528,1.7784,1.9121,2.0548])

    hsc_z_slice_weights = []
    for i in range(nbins):
        weights = nz_interpolators[i](hsc_z_slices)
        hsc_z_slice_weights.append(weights / np.sum(weights))
    
    return np.array(hsc_z_slice_weights)                