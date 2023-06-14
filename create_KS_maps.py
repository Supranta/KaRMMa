import sys
import numpy as np
import h5py as h5
import healpy as hp
import torch
from karmma import KarmmaConfig
from karmma.transforms import shear2conv

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

g1_obs = config.data['g1_obs']
g2_obs = config.data['g2_obs']
    
k_KS_maps = []

N_Z_BINS = config.analysis['nbins']

for i in range(N_Z_BINS):  
    print("Getting KS maps for bin %d"%(i+1))
    g1_in = torch.Tensor(g1_obs[i])
    g2_in = torch.Tensor(g2_obs[i])
    k_KS_i = shear2conv(g1_in, g2_in)
    k_KS_maps.append(k_KS_i.numpy())
    
k_KS_arr = np.array(k_KS_maps)    

with h5.File(config.datafile, 'r+') as f:
    f['kappa_KS'] = k_KS_arr
    
np.save(config.io_dir + '/KS_maps.npy', k_KS_arr)