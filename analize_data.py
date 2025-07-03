import sys
import numpy as np
import h5py as h5
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import torch
import matplotlib.pyplot as plt 
import pandas as pd 
from chainconsumer import Chain, ChainConsumer, make_sample
import pyro.poutine as poutine

torch.set_num_threads(16)

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

nside    = config.analysis['nside']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside

N_Z_BINS = config.analysis['nbins']

sigma_e  = config.analysis['sigma_e']
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
sampler = KarmmaSampler(g1_obs, g2_obs, sigma, mask, lmax, gen_lmax, pixwin=pixwin,
                        shift_file=config.shift_file,mean_g_file=config.mean_g_file,ycl_file=config.y_cl_file,thetafid=config.thetafid)

#============================================================
def compute_log_prob(theta,xlm_real, xlm_imag):
    conditioned_model = poutine.condition(sampler.model,
        data={"theta":    theta,
              "xlm_real": xlm_real,
              "xlm_imag": xlm_imag}
        )
    trace = poutine.trace(conditioned_model).get_trace()
    log_prob = trace.log_prob_sum()  
    return log_prob
#============================================================


io_dir = config.io_dir

N_samples = config.n_samples
steps = np.arange(N_samples)
params = np.zeros((N_samples,config.N_theta+1))

with h5.File(io_dir+f'/mcmc_metadata.h5', 'r') as f:
    int_steps = f['num_steps'][()]

for j in range(N_samples):
    with h5.File(io_dir+f'/sample_{j}.h5', 'r') as f:
        print(f'Sample {j}')
        theta  = torch.tensor(f['theta'][()], dtype=torch.double)
        x_real = torch.tensor(f['xlm_real'][()], dtype=torch.double)
        x_imag = torch.tensor(f['xlm_imag'][()], dtype=torch.double)
        params[j,0] = f['theta'][()][0]
        params[j,1] = f['theta'][()][1]
        params[j,2] = f['theta'][()][2]
        params[j,3] = f['theta'][()][3]
        params[j,4] = f['theta'][()][4]
        params[j,5] = f['theta'][()][5]
        params[j,6] = compute_log_prob(theta, x_real, x_imag)

fig, axes = plt.subplots(2,3)
plt.sca(axes[0,0])
plt.plot(steps,params[:,0], alpha=1., linewidth=0.5)
plt.axhline(config.thetafid[0],linestyle='dashed',color='black')
plt.xlabel('Step')
plt.ylabel(r'$\Omega_m$')
plt.ylim([sampler.emu_lower_bound[0],sampler.emu_upper_bound[0]])

plt.sca(axes[0,1])
plt.plot(steps,params[:,1], alpha=1., linewidth=0.5)
plt.axhline(config.thetafid[1],linestyle='dashed',color='black')
plt.xlabel('Step')
plt.ylabel(r'$\sigma_8$')
plt.ylim([sampler.emu_lower_bound[1],sampler.emu_upper_bound[1]])

plt.sca(axes[0,2])
plt.plot(steps,params[:,2], alpha=1., linewidth=0.5)
plt.axhline(config.thetafid[2],linestyle='dashed',color='black')
plt.xlabel('Step')
plt.ylabel(r'$w$')
plt.ylim([sampler.emu_lower_bound[2],sampler.emu_upper_bound[2]])

plt.sca(axes[1,0])
plt.plot(steps,params[:,3], alpha=1., linewidth=0.5)
plt.axhline(config.thetafid[3],linestyle='dashed',color='black')
plt.xlabel('Step')
plt.ylabel(r'$\omega_b$')
plt.ylim([sampler.emu_lower_bound[3],sampler.emu_upper_bound[3]])

plt.sca(axes[1,1])
plt.plot(steps,params[:,4], alpha=1., linewidth=0.5)
plt.axhline(config.thetafid[4],linestyle='dashed',color='black')
plt.xlabel('Step')
plt.ylabel(r'$h$')
plt.ylim([sampler.emu_lower_bound[4],sampler.emu_upper_bound[4]])

plt.sca(axes[1,2])
plt.plot(steps,params[:,5], alpha=1., linewidth=0.5)
plt.axhline(config.thetafid[5],linestyle='dashed',color='black')
plt.xlabel('Step')
plt.ylabel(r'$n_s$')
plt.ylim([sampler.emu_lower_bound[5],sampler.emu_upper_bound[5]])

fig.suptitle(f'Chain, integration steps = {int_steps}, Tomographic bins = {N_Z_BINS}')

plt.tight_layout()
plt.show()
plt.savefig(config.io_dir+'/chain_plot.png')
plt.close()

c = ChainConsumer()
dframe = pd.DataFrame(params, columns=[r'$\Omega_m$', r'$\sigma_8$', r'$w$', r'$\omega_b$', r'$h$', r'$n_s$', 'log_posterior'])

c.add_chain(Chain(samples=dframe, name=f"Chain", kde=2))


fig = c.plotter.plot()
plt.savefig(config.io_dir+'/contour_plot.png')