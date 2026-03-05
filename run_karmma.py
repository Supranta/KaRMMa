import sys
import pickle
import numpy as np
import h5py as h5
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig, MODE_QUANTITIES
from karmma.utils import *
import karmma.transforms as trf
import torch

torch.set_num_threads(16)

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

nside    = config.analysis['nside']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside
N_Z_BINS = config.analysis['nbins']
sigma_e  = config.analysis['sigma_e']
pixwin   = config.analysis['pixwin']
#============= Load data =======================
g1_obs = config.data['g1_obs']
g2_obs = config.data['g2_obs']
mask   = config.data['mask']
N      = config.data['N']

assert nside==hp.npix2nside(mask.shape[0]), 'Problem with nside!'

sigma = sigma_e / np.sqrt(N + 1e-25)

#============================================================

print("Initializing sampler....")
sampler = KarmmaSampler(g1_obs, g2_obs, sigma, mask, lmax, gen_lmax, pixwin=pixwin,
                        td_file=config.td_file,mode=config.GN_mode,thetafid=config.thetafid,prior_specs=config.prior_specs)
     
print("Done initializing sampler....")

samples, mcmc_kernel = sampler.sample(config.n_burn_in, config.n_samples, config.step_size, x_init=config.x_init)

def x2kappa(xlm_real, xlm_imag, theta_samples_i):
    theta_parts = []
    for i, spec in enumerate(sampler.prior_specs):
        if spec['type'] == 'deterministic':
            theta_parts.append(torch.tensor(spec['value'], dtype=torch.double))
        else:
            theta_parts.append(theta_samples_i[f'theta_{i}'])
    theta = torch.stack(theta_parts)

    kappa_list = []
    xlm    = sampler.get_xlm(xlm_real, xlm_imag)
    cl_key = 'cl_NG' if config.GN_mode == 1 else 'cl_G'
    y_cl   = sampler.emulators[cl_key].predict(theta).reshape((1, N_Z_BINS, N_Z_BINS, -1))[0]
    ylm    = sampler.apply_cl(xlm, y_cl)
    params = {qty: sampler.emulators[qty].predict(theta)
              for qty in MODE_QUANTITIES[config.GN_mode] if qty != cl_key}

    for i in range(N_Z_BINS):
        x = trf.Alm2Map.apply(ylm[i], nside, gen_lmax)
        k = sampler.compute_k(x, i, params)
        k = k.detach().numpy()
        k_filtered = get_filtered_map(k, sampler.pixwin_ell_filter.numpy(), nside)
        kappa_list.append(k_filtered)
    return np.array(kappa_list)

print("Saving samples...")
for i in range(len(samples['xlm_real'])):
    theta_i = {k: samples[k][i] for k in samples if k.startswith('theta_')}
    xlm_real = samples['xlm_real'][i]
    xlm_imag = samples['xlm_imag'][i]
    
    with h5.File(config.io_dir + '/sample_%d.h5' % i, 'w') as f:
        f['i']        = i
        # We reconstruct and store theta as a flat array (as in prev version)
        sorted_keys = sorted(theta_i.keys(), key=lambda k: int(k.split('_')[1]))
        theta_flat  = torch.stack([theta_i[k] for k in sorted_keys]).detach().numpy()
        f['theta'] = theta_flat
        if config.store_fields:
            kappa = x2kappa(xlm_real, xlm_imag, theta_i)
            f['kappa'] = kappa
            f['xlm_real'] = xlm_real
            f['xlm_imag'] = xlm_imag
print("Saving MCMC meta-data and mass matrix...")
with h5.File(config.io_dir + '/mcmc_metadata.h5', 'w') as f:
    f['step_size'] = mcmc_kernel.step_size
    f['num_steps'] = mcmc_kernel.num_steps
with open(config.io_dir + "/mass_matrix_inv.pkl","wb") as f:
    pickle.dump(mcmc_kernel.inverse_mass_matrix, f)
