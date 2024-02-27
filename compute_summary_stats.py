import sys
import numpy as np
import h5py as h5
from karmma import KarmmaConfig
from karmma.utils import *
from mpi4py import MPI

configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

try:
    calculate_cross_corr = bool(sys.argv[2])
    if(calculate_cross_corr):
        print("WILL COMPUTE CROSS-CORRELATION!!")
except:
    calculate_cross_corr = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("rank: %d, size: %d"%(rank, size))

rad2arcmin = 180. / np.pi * 60.

nside = config.analysis['nside']
nbins = config.analysis['nbins']
mask = config.data['mask']

with h5.File(config.io_dir + '/sample_0.h5', 'r') as f:
    kappa_0 = f['kappa'][:]
KAPPA_STD_TRUE = kappa_0[:,mask.astype(bool)].std(1)    

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
        bins       = theta_bins * rad2arcmin
        bin_centre = theta_bin_centre * rad2arcmin
    elif(summary_type=='kappa_pdf'):
        summary    = get_1ptfunc(kappa, kappa_bins, mask) 
        bins       = kappa_bins
        bin_centre = 0.5 * (kappa_bins[:,1:] + kappa_bins[:,:-1])
    elif(summary_type=='peak_counts'):
        summary    = get_tomo_counts(kappa, kappa_bins, mask, flag='peak')
        bins       = kappa_bins
        bin_centre = 0.5 * (kappa_bins[:,1:] + kappa_bins[:,:-1])
    elif(summary_type=='void_counts'):
        summary    = get_tomo_counts(kappa, kappa_bins, mask, flag='void')
        bins       = kappa_bins
        bin_centre = 0.5 * (kappa_bins[:,1:] + kappa_bins[:,:-1])
    elif(summary_type=='pseudo_cl'):
        summary    = get_pseudo_cls(kappa, mask, nmt_ell_bins)
        bins       = ell_bins
        bin_centre = effective_ell
    elif(summary_type=='cross_corr'):
        summary    = get_cross_corr(kappa, kappa_true, mask, theta_bins)
        bins       = theta_bins * rad2arcmin
        bin_centre = theta_bin_centre * rad2arcmin
    return summary, bins, bin_centre

iterations_per_process = config.n_samples // size
start_index = rank * iterations_per_process
end_index = start_index + iterations_per_process

for i in range(start_index,end_index):
    print("i: %d"%(i))
    with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'r+') as f:
        kappa = f['kappa'][:]
    get_summary(i, kappa, 'corr')
    get_summary(i, kappa, 'pseudo_cl')
    get_summary(i, kappa, 'kappa_pdf')
    get_summary(i, kappa, 'peak_counts')
    get_summary(i, kappa, 'void_counts')
    if(calculate_cross_corr):
        get_summary(i, kappa, 'cross_corr')
