import sys
import numpy as np
import h5py as h5
from karmma import KarmmaConfig
from karmma.utils import *
# import torch

# torch.set_num_threads(8)

configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

try:
    calculate_cross_corr = bool(sys.argv[2])
    if(calculate_cross_corr):
        print("WILL COMPUTE CROSS-CORRELATION!!")
except:
    calculate_cross_corr = False
    
nside = config.analysis['nside']
nbins = config.analysis['nbins']
mask = config.data['mask']

with h5.File(config.datafile, 'r') as f:
    kappa_true = f['kappa'][:]
KAPPA_STD_TRUE = kappa_true[:,mask.astype(bool)].std(1)    

theta_bins, theta_bin_centre          = get_theta_bins(nside)
kappa_bins                            = get_kappa_bins_1ptfunc(KAPPA_STD_TRUE, nbins)
nmt_ell_bins, ell_bins, effective_ell = get_nmt_ell_bins(nside)

def get_summary(sample_id, kappa, summary_type):
    if sample_id is None:
        datafile = config.datafile
    else:
        datafile = config.io_dir + '/sample_%d.h5'%(i)
    with h5.File(datafile, 'r+') as f:
        summary_calculated = (summary_type in f)
    if not summary_calculated:
        summary, bins, bin_centre = compute_summary(kappa, mask, summary_type)
        with h5.File(datafile, 'r+') as f:
            summary_grp = f.create_group(summary_type)
            summary_grp['summary']    = summary
            summary_grp['bins']       = bins
            summary_grp['bin_centre'] = bin_centre
    else:
        print(summary_type + ' ALREADY COMPUTED!')        

def compute_summary(kappa, mask, summary_type):
    if(summary_type=='corr'):
        summary    = get_corrfunc(kappa, mask, theta_bins)
        bins       = theta_bins
        bin_centre = theta_bin_centre
    elif(summary_type=='kappa_pdf'):
        summary    = get_1ptfunc(kappa, kappa_bins) 
        bins       = kappa_bins
        bin_centre = 0.5 * (kappa_bins[1:] + kappa_bins[:-1])
    elif(summary_type=='pseudo_cl'):
        summary    = get_pseudo_cls(kappa, mask, nmt_ell_bins)
        bins       = ell_bins
        bin_centre = effective_ell
    elif(summary_type=='cross_corr'):
        summary    = get_cross_corr(kappa, kappa_true, mask, theta_bins)
        bins       = theta_bins
        bin_centre = theta_bin_centre
    return summary, bins, bin_centre
         
with h5.File(config.datafile, 'r+') as f:
    kappa = f['kappa'][:]
    get_summary(None, kappa, 'corr')
    get_summary(None, kappa, 'kappa_pdf')
    get_summary(None, kappa, 'pseudo_cl')
    
for i in range(config.n_samples):
    print("i: %d"%(i))
    with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'r+') as f:
        kappa = f['kappa'][:]
    get_summary(i, kappa, 'corr')
    get_summary(i, kappa, 'kappa_pdf')
    get_summary(i, kappa, 'pseudo_cl')
    if(calculate_cross_corr):
        get_summary(i, kappa, 'cross_corr')