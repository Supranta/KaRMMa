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
            
    def set_config_analysis(self, config_args_analysis):
        print("Setting config data....")
        nbins = int(config_args_analysis['nbins'])
        nside = int(config_args_analysis['nside'])
        sigma_e = float(config_args_analysis['sigma_e'])
        
        split_shift = config_args_analysis['shift'].split(',')
        shift = np.array([float(shift) for shift in split_shift])
        
        split_vargauss = config_args_analysis['vargauss'].split(',')
        vargauss = np.array([float(vargauss) for vargauss in split_vargauss])
        
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
