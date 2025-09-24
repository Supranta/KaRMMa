import sys
import pickle
import numpy as np
import h5py as h5
import healpy as hp
from karmma.karmma_mocks import KarmmaSampler
from karmma.utils import *
import karmma.transforms as trf
import torch
from scipy.stats import qmc
from scipy.special import eval_legendre
from joblib import Parallel, delayed

torch.set_num_threads(16)
#==================================
def compute_lognorm_cl_at_ell(mu, w, integrand, ell):
    xi_g = np.log(np.polynomial.legendre.legval(mu, integrand) + 1)
    return 2 * np.pi * np.sum(w * xi_g * eval_legendre(ell, mu))

def compute_lognorm_cl(cl,nside,nbins,shift,mu,order=2):
    gen_lmax = 3*nside-1
    mu, w = np.polynomial.legendre.leggauss(order * gen_lmax)
    gauss_mu = mu        
    y_cl = np.zeros_like(cl)
    print("Computing y_cl...")
    for i in range(nbins):    
        for j in range(i+1):
            print("z-bin i: %d, j: %d"%(i,j))
            integrand = ((2 * np.arange(gen_lmax + 1) + 1) * cl[i,j] / (4 * np.pi * shift[i] * shift[j]))
            ycl_ij = np.array(Parallel(n_jobs=-1)(
        delayed(compute_lognorm_cl_at_ell)(mu, w, integrand, ell) for ell in range(gen_lmax + 1)))
            y_cl[i,j] = ycl_ij
            y_cl[j,i] = ycl_ij
            
    y_cl[:,:,:2]  = np.tile(1e-20 * np.eye(nbins)[:,:,np.newaxis], (1,1,2))
    return y_cl
# ============================================================================
# We set a "reasonable" prior range. the first 4 entrances are mu and the 
# last four are shift. 

Nsamps = 50
sampler = qmc.LatinHypercube(d=8)
sample = sampler.random(n=Nsamps)
l_bounds = [-6.5,-5.25,-4.5,-4.0,0.0005,0.005,0.005,0.010] 

u_bounds = [-4.0,-3.8,-3.0,-2.5,0.012,0.0225,0.045,0.065]

lnparms_td = qmc.scale(sample, l_bounds, u_bounds)
    
# ===== At the moment, we are just working with one of the GWSt. simulations (id 44).

idx = np.where(np.load('/spiff/ivanespinoza/weak_lensing_data_emulator_data/simulations_run/GWst_good_mock_id_updated.npy') ==44)
_,cl = torch.load('/spiff/ivanespinoza/weak_lensing_data_emulator_data/simulations_run/biased_ccl_cl_good_Cosmo.pt')
cl = cl[idx].numpy()[0]

# ====================================================================================

nside    = 256
gen_lmax = 3 * nside - 1
lmax     = 2 * nside
N_Z_BINS = 4
sigma_e  = 0.261
pixwin   = np.load('/spiff/ivanespinoza/weak_lensing_data_emulator_data/simulations_run/pixwin/pixwin_256.npy')
mask     = hp.fitsfunc.read_map('/spiff/ivanespinoza/weak_lensing_data_emulator_data/simulations_run/mask_desy3.fits')

ycl_td = np.zeros((Nsamps,N_Z_BINS,N_Z_BINS,gen_lmax+1))
for i in range(Nsamps):
    print(f'Working on {i}')
    ycl_td[i] = compute_lognorm_cl(cl,nside,N_Z_BINS,lnparms_td[i,4:],lnparms_td[i,:4])

td = torch.tensor(lnparms_td,dtype=torch.double),torch.tensor(ycl_td,dtype=torch.double)
#============= Load data =======================
with h5.File('/spiff/ivanespinoza/weak_lensing_data_emulator_data/simulations_run_2/karmma_run_3/data.h5', 'r') as f:
    print(f.keys())
    N         = f['N'][()]
    g1_obs    = f['g1_obs'][()]
    g2_obs    = f['g2_obs'][()]

sigma = sigma_e / np.sqrt(N + 1e-25)
#============================================================

print("Initializing sampler....")
sampler = KarmmaSampler(g1_obs, g2_obs, sigma, mask, lmax, gen_lmax, pixwin=pixwin,
                        shift_file='',mean_g_file='',ycl_file=td,thetafid=np.array([-5.2,-4.32,-3.6,-3.2,0.006,0.014,0.028,0.042]))
     
print("Done initializing sampler....")

samples, _ = sampler.sample(200, 200, 0.2, x_init=None,MP=False)

print("Saving samples...")
for i, (theta, xlm_real, xlm_imag) in enumerate(zip(samples['theta'], samples['xlm_real'], samples['xlm_imag'])):
    with h5.File('/spiff/ivanespinoza/weak_lensing_data_emulator_data/td_eduardo_approach' + '/sample_%d.h5'%(i), 'w') as f:
        f['theta']    = theta

