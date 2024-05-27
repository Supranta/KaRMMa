import sys
import numpy as np
import h5py as h5
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import karmma.transforms as trf
from scipy.stats import norm, poisson
import pyro
import torch

torch.set_num_threads(16)

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

nside    = config.analysis['nside']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside - 1

N_Z_BINS = config.analysis['nbins']
shift    = config.analysis['shift']
vargauss = config.analysis['vargauss']

#nz_data  = config.analysis['nz']
sigma_e  = config.analysis['sigma_e']

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
sampler = KarmmaSampler(g1_obs, g2_obs, sigma, mask, cl, shift, vargauss, lmax, gen_lmax, pixwin=pixwin)
print("Done initializing sampler....")

xlm_real = 0.1 * torch.randn((sampler.N_Z_BINS, (sampler.ell > 1).sum()), dtype=torch.double)
xlm_imag = 0.1 * torch.randn(sampler.N_Z_BINS, ((sampler.ell > 1) & (sampler.emm > 0)).sum(), dtype=torch.double)

xlm_real.requires_grad_()
xlm_imag.requires_grad_()

# Set up the optimizer
optim = torch.optim.Adam([xlm_real, xlm_imag], lr=0.03)

losses = []

def x2kappa(xlm_real, xlm_imag):
    kappa_list = []
    xlm = sampler.get_xlm(xlm_real, xlm_imag)
    ylm = sampler.apply_cl(xlm, sampler.y_cl)
    for i in range(N_Z_BINS):
        k = torch.exp(sampler.mu[i] + trf.Alm2Map.apply(ylm[i], nside, gen_lmax)) - sampler.shift[i]
        kappa_list.append(k.numpy())
    return np.array(kappa_list)

# Run the optimization for 1000 steps
from tqdm import trange
for step in trange(2000):
    #print("Step: %d"%(step))
    # Compute the log probability of the data given the current values of mu and sigma
    conditioned_model = pyro.condition(sampler.model, data={"xlm_real": xlm_real, "xlm_imag": xlm_imag})
    trace = pyro.poutine.trace(conditioned_model).get_trace()
    log_prob = trace.log_prob_sum()
    
    # Compute the negative log likelihood as the loss
    loss = -log_prob
    losses.append(loss.item())
    # Compute the gradients of the loss with respect to mu and sigma
    loss.backward()
    
    # Take an optimization step
    optim.step()
    
    # Zero the gradients for the next step
    optim.zero_grad()
    
    if(step%5==0):
        kappa_gan_opt = x2kappa(xlm_real.detach(), xlm_imag.detach())
        with h5.File(config.io_dir + '/kappa_MAP.h5', 'w') as f:
            f['xlm_real'] = xlm_real.detach().numpy()
            f['xlm_imag'] = xlm_imag.detach().numpy()
            f['kappa']    = kappa_gan_opt
            f['step']     = step
            f['loss']     = np.array(losses)
