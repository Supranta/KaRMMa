import sys
import numpy as np
import h5py as h5
import pyccl
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig, ClEmu
from karmma.utils import *
import gpytorch
import karmma.transforms as trf
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(8)

configfile = sys.argv[1]
config = KarmmaConfig(configfile)

nside    = config.config_data['nside']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside - 1

N_Z_BINS = config.config_data['nbins']
shift = config.shift

N_Z_FILE = config.config_io['nz_file']
nz_data = np.load(N_Z_FILE)

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

#============= Load data =======================
datafile = config.config_io['datafile']
with h5.File(datafile, 'r') as f:
    g1_obs = f['g1_obs'][:]
    g2_obs = f['g2_obs'][:]
    mask   = f['mask'][:].astype(bool)
    N      = f['N'][:]

sigma_e = config.config_data['sigma_e']
sigma = sigma_e / np.sqrt(N)
#============================================================
print("Initializing sampler....")
sampler = KarmmaSampler(g1_obs, g2_obs, sigma, mask, cl, shift, cl_emu, lmax, gen_lmax)
print("Done initializing sampler....")

num_burn = config.config_mcmc['n_burn_in']
num_samps = config.config_mcmc['n_samples']
samples = sampler.sample(num_burn, num_samps)
output_dir = config.config_io['output_dir']

def x2kappa(xlm_real, xlm_imag):
    kappa_list = []
    xlm = sampler.get_xlm(xlm_real, xlm_imag)
    ylm = sampler.apply_cl(xlm, sampler.y_cl)
    for i in range(N_Z_BINS):
        k = torch.exp(sampler.mu[i] + trf.Alm2Map.apply(ylm[i], nside, gen_lmax)) - sampler.shift[i]
        kappa_list.append(k.numpy())
    return np.array(kappa_list)

for i, (theta, xlm_real, xlm_imag) in enumerate(zip(samples['theta'], samples['xlm_real'], samples['xlm_imag'])):
    kappa = x2kappa(xlm_real, xlm_imag)
    with h5.File(output_dir + '/sample_%d.h5'%(i), 'w') as f:
        f['i']        = i
        f['theta']    = theta
        f['xlm_real'] = xlm_real
        f['xlm_imag'] = xlm_imag
        f['kappa']    = kappa
