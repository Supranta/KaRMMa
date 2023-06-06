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
        
    def set_config_analysis(self, config_args_analysis):
        print("Setting config data....")
        nbins = int(config_args_analysis['nbins'])
        nside = int(config_args_analysis['nside'])
        sigma_e = float(config_args_analysis['sigma_e'])
        
        split_shift = config_args_analysis['shift'].split(',')
        shift = np.array([float(shift) for shift in split_shift])
        
        split_nbar = config_args_analysis['nbar'].split(',')
        nbar = np.array([float(nbar) for nbar in split_nbar])
        
        nz = np.load(config_args_analysis['nz_file'])
        cl = np.load(config_args_analysis['cl_file'])
        
        data_dict = {'nbins': nbins, 
                     'nside': nside, 
                     'sigma_e': sigma_e, 
                     'shift': shift,
                     'nbar': nbar,
                     'nz': nz,
                     'cl': cl
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