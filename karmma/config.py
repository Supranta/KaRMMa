import numpy as np
import h5py as h5
import yaml

class KarmmaConfig:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        self.data = self.set_config_data(config_args['data'])
        
        self.config_io   = config_args['io']
#         print(self.config_io)
        
    def set_config_data(self, config_args_data):
        print("Setting config data....")
        nbins = int(config_args_data['nbins'])
        nside = int(config_args_data['nside'])
        sigma_e = float(config_args_data['sigma_e'])
        split_shift = config_args_data['shift'].split(',')
        shift = np.array([float(shift) for shift in split_shift])
        
        data_dict = {'nbins': nbins, 
                     'nside': nside, 
                     'sigma_e': sigma_e, 
                     'shift': shift
                    }

        data_dict.update(self.read_data(config_args_data))
#         print(data_dict)
        return data_dict
    
    def read_data(self, config_data):
        with h5.File(config_data['datafile'], 'r') as f:
            N      = f['N'][:]
            g1_obs = f['g1_obs'][:]
            g2_obs = f['g2_obs'][:]
            mask   = f['mask'][:]
        
        nz = np.load(config_data['nz_file'])        
        
        data_dict = {'mask': mask,
                     'g1_obs': g1_obs,
                     'g2_obs': g2_obs,
                     'N': N,
                     'nz': nz
                    }
        return data_dict