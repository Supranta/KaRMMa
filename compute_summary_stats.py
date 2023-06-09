import sys
import numpy as np
import h5py as h5
from karmma import KarmmaConfig
from karmma.utils import *
# import torch

# torch.set_num_threads(8)

configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

nside = config.analysis['nside']
nbins = config.analysis['nbins']
mask = config.data['mask']

with h5.File(config.datafile, 'r') as f:
    kappa_true = f['kappa'][:]
KAPPA_STD_TRUE = kappa_true[:,mask.astype(bool)].std(1)    

theta_bins, theta_bin_centre = get_theta_bins(nside)
kappa_bins = get_kappa_bins_1ptfunc(KAPPA_STD_TRUE, nbins)

def compute_corr(sample_id, kappa):
    with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'r+') as f:
        corr_calculated = ('corr' in f)
    if not corr_calculated:        
        corr = get_corrfunc(kappa, mask, theta_bins)    
        with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'r+') as f:
            corr_grp = f.create_group("corr")
            corr_grp['corr'] = corr
            corr_grp['theta_bins'] = theta_bins
            corr_grp['theta_bin_centre'] = theta_bin_centre
    else:
        print('CORRELATION FUNCTION ALREADY COMPUTED!')

def compute_1pt_pdf(sample_id, kappa):
    with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'r+') as f:
        pdf_calculated = ('kappa_pdf' in f)
    if not pdf_calculated:        
        kappa_pdf = compute_1ptfunc(kappa, kappa_bins) 
        with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'r+') as f:
            kappa_pdf_grp = f.create_group("kappa_pdf")
            kappa_pdf_grp['pdf']              = kappa_pdf
            kappa_pdf_grp['kappa_bins']       = kappa_bins
            kappa_pdf_grp['kappa_bin_centre'] = 0.5 * (kappa_bins[1:] + kappa_bins[:-1])
    else:
        print('1 PT PDF ALREADY COMPUTED!')
        
for i in range(config.n_samples):
    print("i: %d"%(i))
    with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'r+') as f:
        kappa = f['kappa'][:]
    compute_corr(i, kappa)
    compute_1pt_pdf(i, kappa)
            