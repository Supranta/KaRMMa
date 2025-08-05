import numpy as np
import healpy as hp
import h5py as h5
import pickle
import yaml
import os
import torch
from .gansky import *

class KarmmaConfig:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        self.analysis = self.set_config_analysis(config_args['analysis'])
        self.set_config_io(config_args['io'])
        self.set_config_mcmc(config_args['mcmc'])
        if 'gan' in config_args:
            self.set_config_gan(config_args['gan'])
        else:
            self.gen       = None
            self.kappa_std = None

    def set_config_analysis(self, config_args_analysis):
        print("Setting config data....")
        self.nbins = int(config_args_analysis['nbins'])
        self.nside = int(config_args_analysis['nside'])
        sigma_e = float(config_args_analysis['sigma_e'])
        
        lognorm_params_file = config_args_analysis['lognorm_params_file']
        lognorm_params = pickle.load(open(lognorm_params_file, 'rb'))

        y_cl_1d = lognorm_params['y_cl']
        y_cl    = self.reshape_y_cl(y_cl_1d, self.nbins)
        shift = lognorm_params['shift']
        mu    = lognorm_params['mu']

        try:
            pixwin = np.load(config_args_analysis['pixwin'])
            print("USING EMPIRICAL WINDOW FUNCTION!")
        except:
            pixwin='healpix'

        data_dict = {'nbins': self.nbins, 
                     'nside': self.nside, 
                     'sigma_e': sigma_e, 
                     'shift': shift,
                     'mu': mu,
                     'ycl': y_cl,
                     'pixwin': pixwin
                    }

        return data_dict
    
    def set_config_io(self, config_args_io):
        self.datafile = config_args_io['datafile']
        try:
            self.data     = self.read_data(self.datafile)
        except:
            if not os.path.exists(self.datafile):
                print("DATAFILE NOT FOUND!")            
            else:
                print("Error while reading datafile!")
                raise
        self.io_dir   = config_args_io['io_dir']
        try:
            self.maskfile = config_args_io['maskfile']
        except:
            self.maskfile = None
        try:
            with h5.File(config_args_io['x_init_file'], 'r') as f:
                xlm_imag_init = f['xlm_imag'][:]
                xlm_real_init = f['xlm_real'][:]
                self.x_init = [xlm_real_init, xlm_imag_init]
            print("Initialized from file: "+config_args_io['x_init_file'])
        except:
            print("Initialization file not found. Initializing with prior.")
            self.x_init = None

    def read_data(self, datafile):
        with h5.File(datafile, 'r') as f:
            N      = f['N'][:]
            g1_obs = f['g1_obs'][:]
            g2_obs = f['g2_obs'][:]
            mask   = f['mask'][:]
        
        return {'mask': mask,
                'g1_obs': g1_obs,
                'g2_obs': g2_obs,
                'N': N}
    
    def set_config_mcmc(self, config_args_mcmc):
        self.n_burn_in = config_args_mcmc['n_burn_in']
        self.n_samples = config_args_mcmc['n_samples']
        try:
            self.step_size = float(config_args_mcmc['step_size'])
        except:
            self.step_size = 0.05
        try:
            if config_args_mcmc['custom_mass_matrix']:
                with open(self.io_dir+'/mass_matrix_inv.pkl', 'rb') as f:
                    self.inv_mass_matrix = pickle.load(f)
                print("Using custom mass matrix...")
        except:
            self.inv_mass_matrix = None

    def set_config_gan(self, config_args_gan):
        self.gan_path = config_args_gan['model']
        num_channels  = config_args_gan['num_channels']
        num_layers    = config_args_gan['num_layers']

        split_std      = config_args_gan['kappa_std'].split(',')
        self.kappa_std = np.array([float(split_std[i]) for i in range(self.nbins)])

        ones_mask = np.ones(hp.nside2npix(self.nside), dtype=bool)
        device='cuda'
        avg_mat = compute_avg_mat(self.nside, ones_mask, True).to(device)
        self.gen = Generator(self.nbins, avg_mat, num_channels=num_channels, num_layers=num_layers).to(device)
        ckpt = torch.load(self.gan_path, map_location=device)
        
        self.gen.load_state_dict(ckpt['gen_ema'])
        self.gen = self.gen.double()

    def reshape_y_cl(self, y_cl, nbins):
        N_ell = y_cl.shape[1]
        y_cl_reshaped = np.zeros((nbins, nbins, N_ell))
        ind = 0
        for j in range(nbins):
            for k in range(j, nbins):
                ycl_jk = y_cl[ind]
                y_cl_reshaped[j,k] = ycl_jk
                y_cl_reshaped[k,j] = ycl_jk
                ind += 1
        return y_cl_reshaped

