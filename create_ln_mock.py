import sys
import numpy as np
import h5py as h5
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import karmma.transforms as trf
from scipy.stats import norm, poisson
import torch

torch.set_num_threads(8)

configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

try:
    lowpass_filter = bool(sys.argv[2])
    if(lowpass_filter):
        print("CREATING LOW-PASS FILTERED MAPS!")
except:
    lowpass_filter = False

nside    = config.analysis['nside']
nbins    = config.analysis['nbins']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside - 1

# N_Z_BINS = config.analysis['nbins']
shift    = config.analysis['shift']

nz_data  = config.analysis['nz']
sigma_e  = config.analysis['sigma_e']

cl = config.analysis['cl'][:,:,:(gen_lmax + 1)]
cl_emu = None

#============================================================
print("Initializing sampler....")
tmp = np.zeros((nbins,hp.nside2npix(nside)))
tmp = KarmmaSampler(tmp, tmp, tmp, tmp, cl, shift, cl_emu, lmax, gen_lmax)
print("Done initializing sampler....")

ell, emm = hp.Alm.getlm(gen_lmax)

def eigvec_matmul(A, x):
    y = np.zeros_like(x)
    for i in range(nbins):
        for j in range(nbins):
            y[i] += A[i,j] * x[j]
    return y

def apply_cl(xlm, cl):
    L = np.linalg.cholesky(cl.T).T
    
    xlm_real = xlm.real
    xlm_imag = xlm.imag
    
    L_arr = np.swapaxes(L[:,:,ell[ell > -1]], 0,1)
    
    ylm_real = eigvec_matmul(L_arr, xlm_real) / np.sqrt(2.)
    ylm_imag = eigvec_matmul(L_arr, xlm_imag) / np.sqrt(2.)

    ylm_real[:,ell[emm==0]] *= np.sqrt(2)
    
    return ylm_real + 1j * ylm_imag

def get_xlm(xlm_real, xlm_imag):
    ell, emm = hp.Alm.getlm(gen_lmax)
    #==============================
    _xlm_real = np.zeros((nbins, len(ell)))
    _xlm_imag = np.zeros_like(_xlm_real)
    _xlm_real[:,ell > 1] = xlm_real
    _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
    xlm = _xlm_real + 1j * _xlm_imag
    #==============================
    return xlm
    
def generate_xlm():
    xlm_real = np.random.normal(size=(nbins, (ell > 1).sum()))
    xlm_imag = np.random.normal(size=(nbins, ((ell > 1) & (emm > 0)).sum()))

    xlm = get_xlm(xlm_real, xlm_imag)
    return xlm

def generate_mock_y_lm():
    xlm = generate_xlm()
    return apply_cl(xlm, tmp.y_cl)

mask    = hp.fitsfunc.read_map(config.maskfile)
boolean_mask = mask.astype(bool)

def get_y_maps():
    y_lm = generate_mock_y_lm()
    y_maps = []
    for i in range(nbins):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)    
    return np.array(y_maps)    

def get_LN_shear(y_maps):
    g1_list = []
    g2_list = []
    k_list = []
    for i in range(nbins):
        k = np.exp(y_maps[i] + tmp.mu[i]) - shift[i]
        if(lowpass_filter):
            lowpass_ell_filter = (ell < lmax)
            k = get_filtered_map(k, lowpass_ell_filter, nside)
        k_list.append(k)
        g1, g2 = trf.conv2shear(torch.tensor(k), lmax)
        g1 = g1.numpy() * mask
        g2 = g2.numpy() * mask
        g1_list.append(g1)
        g2_list.append(g2)    

    g1 = np.array(g1_list)
    g2 = np.array(g2_list)    
    k_arr  = np.array(k_list)
    
    return g1, g2, k_arr  
        
y_maps            = get_y_maps()
g1, g2, k_arr     = get_LN_shear(y_maps)
g1_obs, g2_obs, N = get_mock_data(nside, nbins, [config.analysis['nbar'], config.analysis['sigma_e'], g1, g2, mask])        
save_datafile(config.datafile, g1_obs, g2_obs, k_arr, N, mask)
