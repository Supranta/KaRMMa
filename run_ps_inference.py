import sys
import numpy as np
import h5py as h5
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import karmma.transforms as trf
from scipy.stats import norm, poisson
import torch
from tqdm import trange

torch.set_num_threads(8)

configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

nside    = config.analysis['nside']
nbins    = config.analysis['nbins']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside 
N_Z_BINS = config.analysis['nbins']
pixwin   = config.analysis['pixwin']
mask         = hp.fitsfunc.read_map(config.maskfile)
boolean_mask = mask.astype(bool)
#=== For shape noise ================
sigma_e  = config.analysis['sigma_e']
N        = np.load(config.N_map)
sigma    = sigma_e / np.sqrt(N + 1e-25)
#=====================================
thetafid = torch.tensor(config.thetafid,dtype=torch.double)
#============================================================
print("Initializing sampler....")
tmp = np.zeros((nbins,hp.nside2npix(nside)))
cl_file = './data/des_y3/cl_pyccl_training_data.pt'
tmp = KarmmaSampler(tmp, tmp, tmp, tmp, lmax, gen_lmax,pixwin=pixwin,shift_file=config.shift_file,mean_g_file=config.mean_g_file,ycl_file=cl_file,thetafid=config.thetafid)
print("Done initializing sampler....")

ell, emm = hp.Alm.getlm(gen_lmax)
cl       = tmp.cl_emu.predict(thetafid).reshape((1, N_Z_BINS, N_Z_BINS, -1))[0].numpy()
mean_g   = tmp.mean_g_emu.predict(thetafid)[0].numpy()
shift    = tmp.shift_emu.predict(thetafid)[0].numpy()

ls = np.arange(lmax + 1)

def get_binned_cl(cl, ell_bins):
    w = (1. + 2 * ls)
    cl_binned = []
    for i in range(len(ell_bins) - 1):
        select_ls = (ls > ell_bins[i]) & (ls <= ell_bins[i+1])
        cl_weighted = np.sum((cl * w)[select_ls]) / np.sum(w[select_ls])
        cl_binned.append(cl_weighted)
    return np.array(cl_binned)

def get_cl(theta, ell_bins):
    theta_tensor = torch.tensor(theta)
    cl_pred = tmp.cl_emu.predict(theta_tensor).reshape((1, N_Z_BINS, N_Z_BINS, -1))[0].numpy()
    cl_list = []
    for i in range(N_Z_BINS):
        for j in range(i+1):
            cl_ij = cl_pred[i,j][:(lmax+1)]
            cl_binned = get_binned_cl(cl_ij, ell_bins)
            cl_list.append(cl_binned)
    return np.array(cl_list).flatten()

N_ell_bins = 13
ell_bins   = np.logspace(np.log10(6), np.log10(lmax), N_ell_bins).astype(int)
print(ell_bins)

cl_fid = get_cl(config.thetafid, ell_bins)

def get_cl_cov(N_mocks=650):
    cl_list = []
    for i in range(N_mocks):
        cl = np.load(config.io_dir + '/cl_%d.npy'%(i))
        cl_list.append(cl)
    cl_arr = np.array(cl_list)
    Omega_s = boolean_mask.sum() / len(boolean_mask)
    print("Omega_s: %2.3f"%(Omega_s))
    cl_cov = np.cov(cl_arr.T) / Omega_s
    cl_invcov = np.linalg.inv(cl_cov)
    return cl_cov, cl_invcov

cl_cov, cl_inv_cov = get_cl_cov()
eigvals = np.linalg.eigvals(cl_cov)
# Make sure that the eigen values are positive. If not, run more mocks for the covariance.
print("eigvals: "+str(eigvals))

def log_prior(theta):
    if np.all((theta >= tmp.emu_lower_bound) & (theta <= tmp.emu_upper_bound)):
        return 0.
    else:
        return -np.inf

def log_lkl(theta):
    cl_pred = get_cl(theta, ell_bins)
    delta_cl = (cl_pred - cl_fid)
    return -0.5 * delta_cl @ cl_inv_cov @ delta_cl

def log_prob(theta):
    return log_prior(theta) + log_lkl(theta)

loglkl_fid = log_lkl(config.thetafid)

ndim = len(config.thetafid)  # Number of parameters
nwalkers = 20  # Should be at least 2*ndim, commonly 2-4 times ndim
nsteps = 1000  # Number of steps to run

# Initialize walkers in a small ball around theta_fid
pos_std = 0.01 * (tmp.emu_upper_bound - tmp.emu_lower_bound)  # 1% of parameter range
pos     = config.thetafid + pos_std * np.random.randn(nwalkers, ndim)

import emcee
# Create the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

# Run the chain
print("Running MCMC...")
sampler.run_mcmc(pos, nsteps, progress=True)
np.save(config.io_dir + '/ps_chain.npy', sampler.chain)
