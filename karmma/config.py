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
        self.set_config_multiplicative_bias(config_args.get('multiplicative_bias', None))
        self.set_config_delta_z(config_args.get('delta_z', None))

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
        

        f = np.load(self.td_file)
        self.cosmo_names = [str(n) for n in f['cosmo_param_names']]
        raw_priors = config_args_cosmo.get('priors', None)
        if raw_priors is not None:
            # reorder priors to match the order in the training data
            prior_dict = {p['name']: p for p in raw_priors}
            self.prior_specs = [prior_dict[name] for name in self.cosmo_names]
        else:
            self.prior_specs = None

    def set_config_multiplicative_bias(self, config_args_mb):
        nbins = self.analysis['nbins']
        if config_args_mb is None:
            # Default is no multiplicative bias
            self.mb_specs = [{'bin': i, 'type': 'deterministic', 'value': 0.0} 
                            for i in range(nbins)]
        else:
            assert len(config_args_mb) == nbins, \
                f"Expected {nbins} multiplicative bias specs, got {len(config_args_mb)}"
            self.mb_specs = sorted([dict(spec) for spec in config_args_mb], key=lambda x: x['bin'])

    def set_config_delta_z(self, config_args_dz):
        nbins = self.analysis['nbins']
        if config_args_dz is None:
            # Default: no Δz nuisance (all fixed to zero)
            self.dz_specs = [{'bin': i, 'type': 'deterministic', 'value': 0.0}
                            for i in range(nbins)]
        else:
            assert len(config_args_dz) == nbins, \
                f"Expected {nbins} delta_z specs, got {len(config_args_dz)}"
            self.dz_specs = sorted([dict(spec) for spec in config_args_dz], key=lambda x: x['bin'])
            
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

    def get_mode(self, filepath):
        f = np.load(filepath)
        return f['params_fid'].shape[-1]
    
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

    def set_config_mocks(self, config_args_mocks):
        self.N_theta   = len(self.cosmo_names)
        theta_fid_dict = config_args_mocks['theta_fid']
        self.thetafid  = np.array([theta_fid_dict[name] for name in self.cosmo_names])
        self.mb_init   = np.array(config_args_mocks.get('mb_init', [0.0] * self.analysis['nbins']))
        self.dz_init   = np.array(config_args_mocks.get('deltaz_init', [0.0] * self.analysis['nbins']))
        self.N_map     = config_args_mocks['N_map']