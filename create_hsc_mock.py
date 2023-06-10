import sys
import numpy as np
import h5py as h5
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import torch

torch.set_num_threads(8)

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

mock_id = int(sys.argv[2])

nside    = config.analysis['nside']
nbins    = config.analysis['nbins']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside - 1

N_Z_BINS = config.analysis['nbins']
shift    = config.analysis['shift']

nz_data  = config.analysis['nz']
sigma_e  = config.analysis['sigma_e']

mock = np.load('/spiff/pierfied/Simulations/HSC_Mocks_Full/mocks/mock_%d.npy'%(mock_id))

nz_interpolators = get_nz_interpolators(nz_data, nbins)
hsc_z_slice_weights = get_hsc_z_slice_weights(nz_interpolators, nbins)

def get_nz_convolved_maps(mock, hsc_z_slice_weights, nbins):
    data = []
    for i in range(nbins):
        print("i: %d"%(i))
        data_bin_i = np.sum(hsc_z_slice_weights[i][:,np.newaxis,np.newaxis] * mock, axis=0)
        data.append(data_bin_i)        
    return np.array(data)

nz_convolved_maps = get_nz_convolved_maps(mock, hsc_z_slice_weights, nbins)

def get_downgraded_maps(nz_convolved_maps, nside, nbins):
    k_list = []
    g1_list = []
    g2_list = []
    
    for i in range(nbins):
        k_i = hp.ud_grade(nz_convolved_maps[i,0], nside)
        g1_i = hp.ud_grade(nz_convolved_maps[i,1], nside)
        g2_i = hp.ud_grade(nz_convolved_maps[i,2], nside)
    
        k_list.append(k_i)
        g1_list.append(g1_i)
        g2_list.append(g2_i)
        
    return np.array(k_list), np.array(g1_list), np.array(g2_list)

k, g1, g2 = get_downgraded_maps(nz_convolved_maps, nside, nbins)

mask    = hp.fitsfunc.read_map(config.maskfile)

data              = [config.analysis['nbar'], config.analysis['sigma_e'], g1, g2, mask]
g1_obs, g2_obs, N = get_mock_data(nside, nbins, data)   
save_datafile(config.datafile, g1_obs, g2_obs, k, N, mask)