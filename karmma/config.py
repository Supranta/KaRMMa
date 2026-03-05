import numpy as np
import h5py as h5
import yaml
import os

class KarmmaConfig:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        self.analysis = self.set_config_analysis(config_args['analysis'])
        self.set_config_io(config_args['io'])
        self.set_config_mcmc(config_args['mcmc'])
        self.set_config_cosmo(config_args['cosmology'])
        self.set_config_mocks(config_args['mocks'])  

    def set_config_analysis(self, config_args_analysis):
        print("Setting config data....")
        nbins = int(config_args_analysis['nbins'])
        nside = int(config_args_analysis['nside'])
        sigma_e = float(config_args_analysis['sigma_e'])
              
        try:
            pixwin = np.load(config_args_analysis['pixwin'])
            print("USING EMPIRICAL WINDOW FUNCTION!")
        except:
            pixwin='healpix'

        data_dict = {'nbins': nbins, 
                     'nside': nside, 
                     'sigma_e': sigma_e, 
                     'pixwin': pixwin
                    }

        return data_dict
    
    def set_config_cosmo(self, config_args_cosmo):
        self.td_file = config_args_cosmo['td_file']
        self.GN_mode = self.get_mode(self.td_file)
        print(f'Using G{self.GN_mode} prior on kappa')
        raw_priors = config_args_cosmo.get('priors', None)
        self.prior_specs = raw_priors
        
    def set_config_io(self, config_args_io):
        self.store_fields = config_args_io['store_fields'] 
        self.datafile     = config_args_io['datafile']
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

    def get_mode(self,filepath):
        with h5.File(filepath, 'r') as hf:
            keys = set(hf.keys())

        modes = {
            1: {'cosmo', 'cl_NG'},
            2: {'cosmo', 'alpha', 'beta', 'cl_G'},
            3: {'cosmo', 'a', 'b', 'c', 'cl_G'},
        }

        for mode, expected_keys in modes.items():
            if keys == expected_keys:
                return mode

        raise ValueError(f"Unrecognized file format with keys: {keys}")
    
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

    def set_config_mocks(self,config_args_mocks):
        self.N_theta  = config_args_mocks['N_theta']
        split_theta   = config_args_mocks['theta_fid'].split(',')
        self.thetafid = np.array([float(split_theta[i]) for i in range(self.N_theta)])
        self.N_map    = config_args_mocks['N_map']