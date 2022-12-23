import numpy as np
import yaml

class KarmmaConfig:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        self.config_data = config_args['data']
        self.config_io   = config_args['io']
        self.config_emu  = config_args['emulator']
        self.config_mcmc = config_args['mcmc']
        
        split_shift = self.config_data['shift'].split(',')
        self.shift = np.array([float(shift) for shift in split_shift])