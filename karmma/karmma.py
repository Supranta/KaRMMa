import numpy as np
import healpy as hp
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from .transforms import Alm2Map, conv2shear
from .transforms_tomo import Alm2MapTomoMP, conv2shear_tomo
import pickle
from joblib import Parallel, delayed
from scipy.special import eval_legendre
##==================================
from joblib import Parallel, delayed
##==================================

class KarmmaSampler:
    def __init__(self, g1_obs, g2_obs, sigma_obs, mask, 
                        y_cl, shift, mu, kappa_std=None,
                        lmax=None, gen_lmax=None, pixwin=None, gen=None):
        self.g1_obs = g1_obs       
        self.g2_obs = g2_obs
        self.N_Z_BINS = g1_obs.shape[0]
        self.sigma_obs = sigma_obs
        self.mask      = mask.astype(bool)
        self.y_cl      = y_cl
        self.shift     = shift
        self.mu        = mu
        self.kappa_std = kappa_std

        self.dtype = torch.double

        self.nside = hp.get_nside(self.g1_obs)
        self.lmax = 2 * self.nside if not lmax else lmax
        self.gen_lmax = 3 * self.nside - 1 if not gen_lmax else gen_lmax
        
        self.ell, self.emm = hp.Alm.getlm(self.gen_lmax)
       
        if pixwin is not None:
            print("Using healpix pixel window function.")
            from scipy.interpolate import interp1d

            ell_pixwin, _ = hp.Alm.getlm(self.lmax)
            if pixwin=='healpix':
                pixwin = hp.sphtfunc.pixwin(self.nside, lmax=self.gen_lmax)
            else:
                pixwin = pixwin
            pixwin_interp = interp1d(np.arange(len(pixwin)), pixwin)
            pixwin_ell_filter = pixwin_interp(ell_pixwin)
            self.pixwin_ell_filter = torch.tensor(pixwin_ell_filter)
        else:
            self.pixwin_ell_filter = None
        self.gen = gen
        self.compute_lognorm_cl()

        theta_fid = np.array([0.233, 0.82])[np.newaxis]
        theta_fid = torch.Tensor(theta_fid).to(self.dtype)
        self.y_cl_fid = self.y_cl
        self.tensorize()
    
    def tensorize(self):
        self.g1_obs = torch.tensor(self.g1_obs)
        self.g2_obs = torch.tensor(self.g2_obs)
        self.sigma_obs = torch.tensor(self.sigma_obs)
        self.mask = torch.tensor(self.mask)
        self.y_cl = torch.tensor(self.y_cl)

    def compute_lognorm_cl(self, order=2):
        self.mu_torch    = torch.tensor(self.mu[:,np.newaxis])
        self.shift_torch = torch.tensor(self.shift[:,np.newaxis])
        if self.kappa_std is not None:
            self.kappa_std = torch.tensor(self.kappa_std).unsqueeze(1)
        self.y_cl[:,:,:2]  = np.tile(1e-20 * np.eye(self.N_Z_BINS)[:,:,np.newaxis], (1,1,2))

    def get_xlm(self, xlm_real, xlm_imag):
        ell, emm = hp.Alm.getlm(self.gen_lmax)
        _xlm_real = torch.zeros(self.N_Z_BINS, len(ell), dtype=self.dtype)
        _xlm_imag = torch.zeros_like(_xlm_real)
        _xlm_real[:,ell > 1] = xlm_real
        _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
        xlm = _xlm_real + 1j * _xlm_imag
        return xlm

    def matmul(self, A, x):
        y = torch.zeros_like(x)
        for i in range(self.N_Z_BINS):
            for j in range(self.N_Z_BINS):
                y[i] += A[i,j] * x[j]
        return y

    def apply_cl(self, xlm, cl):
        ell, emm = hp.Alm.getlm(self.gen_lmax)
        
        L = torch.linalg.cholesky(cl.T).T
    
        xlm_real = xlm.real
        xlm_imag = xlm.imag
        
        L_arr = torch.swapaxes(L[:,:,ell[ell > -1]], 0,1)
    

        ylm_real = self.matmul(L_arr, xlm_real) / torch.sqrt(torch.Tensor([2.]))
        ylm_imag = self.matmul(L_arr, xlm_imag) / torch.sqrt(torch.Tensor([2.]))

        ylm_real[:,ell[emm==0]] *= torch.sqrt(torch.Tensor([2.]))
    
        return ylm_real + 1j * ylm_imag
    
    def kln2gan(self, k_ln):
        x_ln = (k_ln / self.kappa_std).unsqueeze(0)
        x_gan = self.gen(x_ln)
        k_gan = (x_gan * self.kappa_std).squeeze(0)
        return k_gan

    def model(self, prior_only=True):
        ell, emm = hp.Alm.getlm(self.gen_lmax)

        xlm_real = pyro.sample('xlm_real', dist.Normal(torch.zeros(self.N_Z_BINS, (ell > 1).sum(), dtype=self.dtype),
                                                       torch.ones(self.N_Z_BINS, (ell > 1).sum(), dtype=self.dtype)))
        xlm_imag = pyro.sample('xlm_imag', dist.Normal(torch.zeros(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=self.dtype),
                                                       torch.ones(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=self.dtype)))
          
        xlm = self.get_xlm(xlm_real, xlm_imag)
        y_cl = self.y_cl
        
        ylm    = self.apply_cl(xlm, y_cl)
        y_maps = Alm2MapTomoMP.apply(ylm, self.nside, self.gen_lmax) + self.mu_torch
        k_ln   = torch.exp(y_maps) - self.shift_torch

        if self.gen is not None:
            k_maps = self.kln2gan(k_ln)
        else:
            k_maps = k_ln

        if not prior_only:
            g1_tomo, g2_tomo = conv2shear_tomo(k_maps, self.lmax, self.pixwin_ell_filter)
            
            for i in range(self.N_Z_BINS):
                pyro.sample(f'g1_obs_{i}', dist.Normal(g1_tomo[i,self.mask], self.sigma_obs[i,self.mask]), obs=self.g1_obs[i,self.mask])
                pyro.sample(f'g2_obs_{i}', dist.Normal(g2_tomo[i,self.mask], self.sigma_obs[i,self.mask]), obs=self.g2_obs[i,self.mask])

    def sample(self, num_burn, num_samples, step_size=0.05, inv_mass_matrix=None, x_init=None):
        kernel = NUTS(self.model, target_accept_prob=0.65, step_size=step_size)
        if inv_mass_matrix is not None:
            kernel.mass_matrix_adapter.inverse_mass_matrix = inv_mass_matrix
        x_real_init = 0.3 * torch.randn((self.N_Z_BINS, (self.ell > 1).sum()), dtype=self.dtype)
        x_imag_init = 0.3 * torch.randn((self.N_Z_BINS, ((self.ell > 1) & (self.emm > 0)).sum()), dtype=self.dtype)
        if x_init is not None:
            xlm_real_init, xlm_imag_init = x_init
            xlm_real_init = torch.tensor(xlm_real_init, dtype=self.dtype)
            xlm_imag_init = torch.tensor(xlm_imag_init, dtype=self.dtype)

        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_burn,
                    initial_params={"xlm_real": x_real_init,
                                    "xlm_imag": x_imag_init})
        mcmc.run()
        self.samps = mcmc.get_samples()

        return self.samps, mcmc.kernel

    def save_samples(self, fname):
        pickle.dump(self.samps, open(fname, 'wb'))
