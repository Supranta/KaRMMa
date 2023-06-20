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

try:
    lowpass_filter = bool(sys.argv[2])
    if(lowpass_filter):
        print("CREATING LOW-PASS FILTERED MAPS!")
except:
    lowpass_filter = False

rad2arcmin = 180. / np.pi * 60.

nside    = config.analysis['nside']
nbins    = config.analysis['nbins']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside - 1

N_Z_BINS = config.analysis['nbins']
shift    = config.analysis['shift']

nz_data  = config.analysis['nz']
sigma_e  = config.analysis['sigma_e']

nz_interpolators = get_nz_interpolators(nz_data, nbins)
hsc_z_slice_weights = get_hsc_z_slice_weights(nz_interpolators, nbins)

def get_nz_convolved_maps(mock, hsc_z_slice_weights, nbins):
    data = []
    for i in range(nbins):
        print("i: %d"%(i))
        data_bin_i = np.sum(hsc_z_slice_weights[i][:,np.newaxis,np.newaxis] * mock, axis=0)
        data.append(data_bin_i)        
    return np.array(data)

def get_downgraded_maps(nz_convolved_maps, nside, nbins, lowpass_filter=False, ell_max_filter=None):
    k_list = []
    g1_list = []
    g2_list = []

    for i in range(nbins):
        print("Creating downgraded maps for bin # %d"%(i+1))
        if(lowpass_filter):
            k_i = get_filtered_map(nz_convolved_maps[i,0], ell_max_filter, nside)
            g1_i = get_filtered_map(nz_convolved_maps[i,1], ell_max_filter, nside)
            g2_i = get_filtered_map(nz_convolved_maps[i,2], ell_max_filter, nside)
        else:
            k_i = hp.ud_grade(nz_convolved_maps[i,0], nside)
            g1_i = hp.ud_grade(nz_convolved_maps[i,1], nside)
            g2_i = hp.ud_grade(nz_convolved_maps[i,2], nside)

        k_list.append(k_i)
        g1_list.append(g1_i)
        g2_list.append(g2_i)

    return np.array(k_list), np.array(g1_list), np.array(g2_list)

mask    = hp.fitsfunc.read_map(config.maskfile)

with h5.File(config.datafile, 'r') as f:
    kappa_true = f['kappa'][:]
    mask_data  = f['mask'][:]

KAPPA_STD_TRUE = kappa_true[:,mask_data.astype(bool)].std(1)    

theta_bins, theta_bin_centre          = get_theta_bins(nside)
kappa_bins                            = get_kappa_bins_1ptfunc(KAPPA_STD_TRUE, nbins)
nmt_ell_bins, ell_bins, effective_ell = get_nmt_ell_bins(nside)

for mock_id in range(108):
    print("=========================")
    print("mock_id: %d"%(mock_id))
    print("=========================")
    mock = np.load('/spiff/pierfied/Simulations/HSC_Mocks_Full/mocks/mock_%d.npy'%(mock_id))
    nz_convolved_maps = get_nz_convolved_maps(mock, hsc_z_slice_weights, nbins)
    if lowpass_filter:
        print("Getting low-pass maps...")
        ell, emm = hp.Alm.getlm(3 * 2048 - 1)
        ell_max_filter = (ell <= 2 * nside).astype(float)
        k, g1, g2 = get_downgraded_maps(nz_convolved_maps, nside, nbins, lowpass_filter, ell_max_filter)
        data              = [config.analysis['nbar'], config.analysis['sigma_e'], -g1, -g2, mask]
        g1_obs, g2_obs, N = get_mock_data(nside, nbins, data)
        datafile = 'data/HSC_mocks/des_y3_mock/all_mocks/low_pass/mock_%d.h5'%(mock_id)
    else:
        print("Getting high-frequency maps...")
        mask      = hp.fitsfunc.read_map(config.maskfile)
        k, g1, g2 = get_downgraded_maps(nz_convolved_maps, 1024, nbins)
        data      = [config.analysis['nbar'], config.analysis['sigma_e'], -g1, -g2, mask]

        g1_obs_hires, g2_obs_hires, N_hires = get_mock_data(1024, nbins, data)    
        nz_convolved_maps_1024 = np.stack([k, g1_obs_hires, g2_obs_hires], axis=1)

        k, g1_obs, g2_obs = get_downgraded_maps(nz_convolved_maps_1024, nside, nbins)    
        N                 = hp.ud_grade(N_hires, nside, power=-2)
        mask = hp.ud_grade(mask, nside)
        
        datafile = 'data/HSC_mocks/des_y3_mock/all_mocks/high_frequency/mock_%d.h5'%(mock_id)
       
    save_datafile(datafile, g1_obs, g2_obs, k, N, mask)