import numpy as np
import healpy as hp
import h5py as h5
import torch
import pyro
from .QuadEmulator import ParamsEmu, ClEmu_Cholesky
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from .transforms import Alm2Map, conv2shear
import pickle

MODE_QUANTITIES = {
    1: ['cl_NG'],
    2: ['alpha', 'beta', 'cl_G'],
    3: ['a', 'b', 'c', 'cl_G'],
}

EMULATOR_CLASS = {
    'cl_NG': ClEmu_Cholesky,
    'cl_G':  ClEmu_Cholesky,
    'alpha': ParamsEmu,
    'beta':  ParamsEmu,
    'a':     ParamsEmu,
    'b':     ParamsEmu,
    'c':     ParamsEmu,
}

class KarmmaSampler:
    def __init__(self, g1_obs, g2_obs, sigma_obs, mask, lmax=None, gen_lmax=None, pixwin=None, 
                td_file=None, mode=None, thetafid=None, prior_specs=None, mb_specs=None, mb_init=None):
        self.g1_obs        = g1_obs       
        self.g2_obs        = g2_obs
        self.N_Z_BINS      = g1_obs.shape[0]
        self.sigma_obs     = sigma_obs
        self.mask          = mask.astype(bool)

        self.nside         = hp.get_nside(self.g1_obs)
        self.lmax          = 2 * self.nside if not lmax else lmax
        self.gen_lmax      = 3 * self.nside - 1 if not gen_lmax else gen_lmax
        self.mode          = mode
        self.ell, self.emm = hp.Alm.getlm(self.gen_lmax)

        self.train_emulator(td_file, mode)
        self.emu_upper_bound = self.emulators['cl_NG' if mode == 1 else 'cl_G'].emu[0].cosmology_max
        self.emu_lower_bound = self.emulators['cl_NG' if mode == 1 else 'cl_G'].emu[0].cosmology_min
        self.prior_specs     = self.build_priors(prior_specs)
        self.mb_specs = mb_specs
        self.mb_init  = mb_init if mb_init is not None else np.zeros(self.N_Z_BINS)
        if pixwin is not None:
            print("Using healpix pixel window function.")
            from scipy.interpolate import interp1d

            ell_pixwin, _ = hp.Alm.getlm(self.lmax)
            if pixwin=='healpix':
                pixwin = hp.sphtfunc.pixwin(self.nside, lmax=self.gen_lmax)
            else:
                pixwin = pixwin
            pixwin_interp          = interp1d(np.arange(len(pixwin)), pixwin)
            pixwin_ell_filter      = pixwin_interp(ell_pixwin)
            self.pixwin_ell_filter = torch.tensor(pixwin_ell_filter)
        else:
            self.pixwin_ell_filter = None

        self.theta_fid = torch.Tensor(thetafid).to(torch.double)
        self.tensorize()
    
    def train_emulator(self, td_file, mode):
        with h5.File(td_file, 'r') as hf:
            cosmo = hf['cosmo'][:]
            data  = {qty: hf[qty][:] for qty in MODE_QUANTITIES[mode]}

        self.emulators = {}
        for qty, values in data.items():
            emu = EMULATOR_CLASS[qty]((cosmo, values))
            emu.fit_params()
            self.emulators[qty] = emu
        del data, cosmo

    def build_priors(self, prior_specs):
        N_params = len(self.emu_lower_bound)
        if prior_specs is None:
            return [{'type': 'uniform'} for _ in range(N_params)]
        
        assert len(prior_specs) == N_params, \
            f"Expected {N_params} prior specs, got {len(prior_specs)}"
        
        resolved = []
        for i, spec in enumerate(prior_specs):
            spec = dict(spec)
            if spec['type'] == 'uniform':
                spec.setdefault('low',  float(self.emu_lower_bound[i]))
                spec.setdefault('high', float(self.emu_upper_bound[i]))
            elif spec['type'] not in ('gaussian', 'deterministic'):
                raise ValueError(f"Unknown prior type: {spec['type']}")
            resolved.append(spec)
        print(resolved)
        return resolved

    def sample_theta(self):
        parts = []
        for i, spec in enumerate(self.prior_specs):
            ptype = spec['type']
            if ptype == 'deterministic':
                parts.append(torch.tensor([spec['value']], dtype=torch.double))
            elif ptype == 'uniform':
                low  = torch.tensor(spec['low'],  dtype=torch.double)
                high = torch.tensor(spec['high'], dtype=torch.double)
                parts.append(pyro.sample(f'theta_{i}', dist.Uniform(low, high)).reshape(1))
            elif ptype == 'gaussian':
                mu    = torch.tensor(spec['mu'],    dtype=torch.double)
                sigma = torch.tensor(spec['sigma'], dtype=torch.double)
                parts.append(pyro.sample(f'theta_{i}', dist.Normal(mu, sigma)).reshape(1))
        return torch.stack(parts).squeeze()
    
    def sample_nuisance(self):
        m = []
        for i, spec in enumerate(self.mb_specs):
            ptype = spec['type']
            if ptype == 'deterministic':
                m.append(torch.tensor(spec['value'], dtype=torch.double))
            elif ptype == 'uniform':
                low  = torch.tensor(spec['low'],  dtype=torch.double)
                high = torch.tensor(spec['high'], dtype=torch.double)
                m.append(pyro.sample(f'm_{i}', dist.Uniform(low, high)))
            elif ptype == 'gaussian':
                mu    = torch.tensor(spec['mu'],    dtype=torch.double)
                sigma = torch.tensor(spec['sigma'], dtype=torch.double)
                m.append(pyro.sample(f'm_{i}', dist.Normal(mu, sigma)))
        return torch.stack(m)
    
    def domain_barrier(self, theta):
        low     = torch.tensor(self.emu_lower_bound, dtype=torch.double)
        high    = torch.tensor(self.emu_upper_bound, dtype=torch.double)
        epsilon = 0.005 * (high - low)
        log_gate = (
            torch.log(torch.sigmoid((theta - low)  / epsilon)) +
            torch.log(torch.sigmoid((high - theta) / epsilon))
        ).sum()
        pyro.factor('domain_barrier', log_gate)

    def tensorize(self):
        self.g1_obs    = torch.tensor(self.g1_obs)
        self.g2_obs    = torch.tensor(self.g2_obs)
        self.sigma_obs = torch.tensor(self.sigma_obs)
        self.mask      = torch.tensor(self.mask)

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
    
    def compute_k(self, x, i, params):
        if self.mode == 1:
            return x
        elif self.mode == 2:
            alpha, beta = params['alpha'][0][i], params['beta'][0][i]
            return beta * torch.exp(alpha * x - 0.5 * alpha**2) - beta
        elif self.mode == 3:
            a, b, c = params['a'][0][i], params['b'][0][i], params['c'][0][i]
            return (torch.exp(a * x - 0.5 * a**2) + b*x + c) / (1 + c) - 1.   
        
    def model(self, prior_only=False):
        ell, emm = hp.Alm.getlm(self.gen_lmax)
        theta    = self.sample_theta()
        self.domain_barrier(theta)
        theta_clamped = torch.clamp(theta,
                            min=torch.tensor(self.emu_lower_bound, dtype=torch.double),
                            max=torch.tensor(self.emu_upper_bound, dtype=torch.double))
        xlm_real = pyro.sample('xlm_real', dist.Normal(torch.zeros(self.N_Z_BINS, (ell > 1).sum(), dtype=torch.double),
                                                    torch.ones(self.N_Z_BINS, (ell > 1).sum(), dtype=torch.double)))
        xlm_imag = pyro.sample('xlm_imag', dist.Normal(torch.zeros(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=torch.double),
                                                    torch.ones(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=torch.double)))
        xlm  = self.get_xlm(xlm_real, xlm_imag)
        cl_key = 'cl_NG' if self.mode == 1 else 'cl_G'
        y_cl = self.emulators[cl_key].predict(theta_clamped).reshape((1, self.N_Z_BINS, self.N_Z_BINS, -1))[0]
        ylm  = self.apply_cl(xlm, y_cl)

        # predict all non-cl quantities for this mode
        params = {qty: self.emulators[qty].predict(theta_clamped) 
                for qty in MODE_QUANTITIES[self.mode] if qty != cl_key}

        m = self.sample_nuisance()
        for i in range(self.N_Z_BINS):
            x      = Alm2Map.apply(ylm[i], self.nside, self.gen_lmax)
            k      = self.compute_k(x, i, params)
            g1, g2 = conv2shear(k, self.lmax, self.pixwin_ell_filter)
            pyro.sample(f'g1_obs_{i}', dist.Normal((1 + m[i]) * g1[self.mask], self.sigma_obs[i,self.mask]), obs=self.g1_obs[i,self.mask])
            pyro.sample(f'g2_obs_{i}', dist.Normal((1 + m[i]) * g2[self.mask], self.sigma_obs[i,self.mask]), obs=self.g2_obs[i,self.mask])

    def sample(self, num_burn, num_samples, step_size=0.05, x_init=None):
        kernel = NUTS(self.model, target_accept_prob=0.65, step_size=step_size)

        x_real_init = 0.3 * torch.randn((self.N_Z_BINS, (self.ell > 1).sum()), dtype=torch.double)
        x_imag_init = 0.3 * torch.randn((self.N_Z_BINS, ((self.ell > 1) & (self.emm > 0)).sum()), dtype=torch.double)

        if x_init is not None:
            x_real_init = torch.tensor(x_init[0], dtype=torch.double)
            x_imag_init = torch.tensor(x_init[1], dtype=torch.double)

        # build theta initial params, skipping deterministic entries
        theta_init = {}
        for i, spec in enumerate(self.prior_specs):
            if spec['type'] != 'deterministic':
                theta_init[f'theta_{i}'] = self.theta_fid[i].unsqueeze(0)

        mb_init = {}
        for i, spec in enumerate(self.mb_specs):
            if spec['type'] != 'deterministic':
                mb_init[f'm_{i}'] = torch.tensor(float(self.mb_init[i]), dtype=torch.double)

        initial_params = {
            **theta_init,
            **mb_init,
            'xlm_real': x_real_init,
            'xlm_imag': x_imag_init,
        }

        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_burn,
                    initial_params=initial_params)
        mcmc.run()
        self.samps = mcmc.get_samples()
        return self.samps, mcmc.kernel

    def save_samples(self, fname):
        pickle.dump(self.samps, open(fname, 'wb'))
