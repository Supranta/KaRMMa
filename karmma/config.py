import numpy as np
import h5py as h5
import pickle
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
            
    def set_config_analysis(self, config_args_analysis):
        print("Setting config data....")
        nbins = int(config_args_analysis['nbins'])
        nside = int(config_args_analysis['nside'])
        sigma_e = float(config_args_analysis['sigma_e'])
        
        split_shift = config_args_analysis['shift'].split(',')
        shift = np.array([float(split_shift[i]) for i in range(nbins)])
        
        split_vargauss = config_args_analysis['vargauss'].split(',')
        vargauss = np.array([float(split_vargauss[i]) for i in range(nbins)])
        
        cl = np.load(config_args_analysis['cl_file'])
       
        try:
            pixwin = np.load(config_args_analysis['pixwin'])
            print("USING EMPIRICAL WINDOW FUNCTION!")
        except:
            pixwin='healpix'

        data_dict = {'nbins': nbins, 
                     'nside': nside, 
                     'sigma_e': sigma_e, 
                     'shift': shift,
                     'vargauss': vargauss,
                     'cl': cl,
                     'pixwin': pixwin
                    }

        return data_dict
    
    def set_config_cosmo(self, config_args_cosmo):
        self.emulator_file = config_args_cosmo['emulator_file']
        
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
