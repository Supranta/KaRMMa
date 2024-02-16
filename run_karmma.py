import sys
import numpy as np
import h5py as h5
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig#, ClEmu
from karmma.utils import *
# import gpytorch
import karmma.transforms as trf
from scipy.stats import norm, poisson
import torch

torch.set_num_threads(8)

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

nside    = config.analysis['nside']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside

N_Z_BINS = config.analysis['nbins']
shift    = config.analysis['shift']
vargauss = config.analysis['vargauss']

nz_data  = config.analysis['nz']
sigma_e  = config.analysis['sigma_e']

"""
emu_file = config.config_emu['emu_file']

with h5.File(emu_file, 'r') as f:
    cl          = f['cl_fid'][:]
    train_theta = f['theta'][:]
    train_cl    = f['y_cl'][:]    

N_PCA = config.config_emu['n_pca']
cl_emu = ClEmu([train_theta, train_cl], N_PCA)

print("Training emulator....")
cl_emu.train_emu()
print("Done training emulator....")
"""
cl     = config.analysis['cl'][:,:,:(gen_lmax + 1)]
cl_emu = None
pixwin = config.analysis['pixwin']

#============= Load data =======================
g1_obs = config.data['g1_obs']
g2_obs = config.data['g2_obs']
mask   = config.data['mask']
N      = config.data['N']

assert nside==hp.npix2nside(mask.shape[0]), 'Problem with nside!'

sigma = sigma_e / np.sqrt(N + 1e-25)

#============================================================

print("Initializing sampler....")
sampler = KarmmaSampler(g1_obs, g2_obs, sigma, mask, cl, shift, vargauss, cl_emu, lmax, gen_lmax, pixwin=pixwin)
     
print("Done initializing sampler....")

samples = sampler.sample(config.n_burn_in, config.n_samples)

def x2kappa(xlm_real, xlm_imag):
    kappa_list = []
    kappa_filtered_list = []
    xlm = sampler.get_xlm(xlm_real, xlm_imag)
    ylm = sampler.apply_cl(xlm, sampler.y_cl)
    for i in range(N_Z_BINS):
        k = torch.exp(sampler.mu[i] + trf.Alm2Map.apply(ylm[i], nside, gen_lmax)) - sampler.shift[i]
        k = k.numpy()
        k_filtered = get_filtered_map(k, sampler.pixwin_ell_filter.numpy(), nside)
        kappa_list.append(k)
        kappa_filtered_list.append(k_filtered)
    return np.array(kappa_list), np.array(kappa_filtered_list)

for i, (theta, xlm_real, xlm_imag) in enumerate(zip(samples['theta'], samples['xlm_real'], samples['xlm_imag'])):
    kappa, kappa_filtered = x2kappa(xlm_real, xlm_imag)
    with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'w') as f:
        f['i']        = i
        f['theta']    = theta
        #f['xlm_real'] = xlm_real
        #f['xlm_imag'] = xlm_imag
        f['kappa']    = kappa_filtered
        #f['kappa'] = kappa
