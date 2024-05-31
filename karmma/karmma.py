import numpy as np
import healpy as hp
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from .transforms import Alm2Map, conv2shear
import pickle
from joblib import Parallel, delayed
from scipy.special import eval_legendre
##==================================
from joblib import Parallel, delayed
##==================================

class KarmmaSampler:
    def __init__(self, g1_obs, g2_obs, sigma_obs, mask, cl, shift, vargauss, lmax=None, gen_lmax=None, pixwin=None):
        self.g1_obs = g1_obs       
        self.g2_obs = g2_obs
        self.N_Z_BINS = g1_obs.shape[0]
        self.sigma_obs = sigma_obs
        self.mask = mask.astype(bool)
        self.cl = cl
        self.shift    = shift
        self.vargauss = vargauss

        self.y_cl     = np.zeros_like(cl)
        
        self.mu = np.zeros(self.N_Z_BINS)

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

        self.compute_lognorm_cl()

        theta_fid = np.array([0.233, 0.82])[np.newaxis]
        theta_fid = torch.Tensor(theta_fid).to(torch.double)
        self.y_cl_fid = self.y_cl
        self.tensorize()
    
    def tensorize(self):
        self.g1_obs = torch.tensor(self.g1_obs)
        self.g2_obs = torch.tensor(self.g2_obs)
        self.sigma_obs = torch.tensor(self.sigma_obs)
        self.mask = torch.tensor(self.mask)
        self.cl = torch.Tensor(self.cl)
        self.y_cl = torch.tensor(self.y_cl)

    def compute_lognorm_cl_at_ell(self, mu, w, integrand, ell):
        integrand = np.log(np.polynomial.legendre.legval(mu, integrand) + 1)
        return 2 * np.pi * np.sum(w * integrand * eval_legendre(ell, mu))

    def compute_lognorm_cl(self, order=2):
        mu, w = np.polynomial.legendre.leggauss(order * self.gen_lmax)
        
        print("Computing mu/sigma2....")
        for i in range(self.N_Z_BINS):           
            self.mu[i] = np.log(self.shift[i]) - 0.5 * self.vargauss[i]            
        
        print("Computing y_cl...")
        for i in range(self.N_Z_BINS):    
            for j in range(i+1):
                print("z-bin i: %d, j: %d"%(i,j))
                integrand = ((2 * np.arange(self.gen_lmax + 1) + 1) * self.cl[i,j] / (4 * np.pi * self.shift[i] * self.shift[j]))

                ycl_ij = np.array(Parallel(n_jobs=-1)(
            delayed(self.compute_lognorm_cl_at_ell)(mu, w, integrand, ell) for ell in range(self.gen_lmax + 1)))
                self.y_cl[i,j] = ycl_ij
                self.y_cl[j,i] = ycl_ij
                
        self.y_cl[:,:,:2]  = np.tile(1e-20 * np.eye(self.N_Z_BINS)[:,:,np.newaxis], (1,1,2))

    def get_xlm(self, xlm_real, xlm_imag):
        ell, emm = hp.Alm.getlm(self.gen_lmax)
        _xlm_real = torch.zeros(self.N_Z_BINS, len(ell), dtype=torch.double)
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
    
    def model(self, prior_only=False):
        ell, emm = hp.Alm.getlm(self.gen_lmax)

        xlm_real = pyro.sample('xlm_real', dist.Normal(torch.zeros(self.N_Z_BINS, (ell > 1).sum(), dtype=torch.double),
                                                       torch.ones(self.N_Z_BINS, (ell > 1).sum(), dtype=torch.double)))
        xlm_imag = pyro.sample('xlm_imag', dist.Normal(torch.zeros(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=torch.double),
                                                       torch.ones(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=torch.double)))
          
        xlm = self.get_xlm(xlm_real, xlm_imag)
        y_cl = self.y_cl
        
        ylm = self.apply_cl(xlm, y_cl)
       
        for i in range(self.N_Z_BINS):
            k = torch.exp(self.mu[i] + Alm2Map.apply(ylm[i], self.nside, self.gen_lmax)) - self.shift[i]
            g1, g2 = conv2shear(k, self.lmax, self.pixwin_ell_filter)

            pyro.sample(f'g1_obs_{i}', dist.Normal(g1[self.mask], self.sigma_obs[i,self.mask]), obs=self.g1_obs[i,self.mask])
            pyro.sample(f'g2_obs_{i}', dist.Normal(g2[self.mask], self.sigma_obs[i,self.mask]), obs=self.g2_obs[i,self.mask])
    
    def sample(self, num_burn, num_samples, inv_mass_matrix=None, x_init=None):
        kernel = NUTS(self.model, target_accept_prob=0.65)
        if inv_mass_matrix is not None:
            kernel.mass_matrix_adapter.inverse_mass_matrix = inv_mass_matrix
        x_real_init = 0.3 * torch.randn((self.N_Z_BINS, (self.ell > 1).sum()), dtype=torch.double)
        x_imag_init = 0.3 * torch.randn((self.N_Z_BINS, ((self.ell > 1) & (self.emm > 0)).sum()), dtype=torch.double)
        if x_init is not None:
            xlm_real_init, xlm_imag_init = x_init
            xlm_real_init = torch.tensor(xlm_real_init, dtype=torch.double)
            xlm_imag_init = torch.tensor(xlm_imag_init, dtype=torch.double)

        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_burn,
                    initial_params={"xlm_real": x_real_init,
                                    "xlm_imag": x_imag_init})
        mcmc.run()
        self.samps = mcmc.get_samples()

        return self.samps, mcmc.kernel

    def save_samples(self, fname):
        pickle.dump(self.samps, open(fname, 'wb'))
