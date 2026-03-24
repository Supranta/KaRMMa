import sys
import numpy as np
import h5py as h5
from karmma import KarmmaSampler, KarmmaConfig, MODE_QUANTITIES
from karmma.utils import *
import karmma.transforms as trf
import torch

torch.set_num_threads(8)

configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

nside        = config.analysis['nside']
gen_lmax     = 3 * nside - 1
lmax         = 2 * nside 
N_Z_BINS     = config.analysis['nbins']
pixwin       = config.analysis['pixwin']
mask         = np.load(config.maskfile)
boolean_mask = mask.astype(bool)
#=== For shape noise ================
sigma_e  = config.analysis['sigma_e']
N        = np.load(config.N_map)
sigma    = sigma_e / np.sqrt(N + 1e-25)
#=====================================
thetafid = torch.tensor(config.thetafid,dtype=torch.double)
#============================================================
print("Initializing sampler....")
tmp = np.zeros((N_Z_BINS,hp.nside2npix(nside)))
tmp = KarmmaSampler(tmp, tmp, tmp, tmp, lmax, gen_lmax, pixwin=pixwin,
                        td_file=config.td_file,mode=config.GN_mode,thetafid=config.thetafid,prior_specs=config.prior_specs, mb_specs=config.mb_specs, mb_init=config.mb_init)
print("Done initializing sampler....")

cl_key   = 'cl_NG' if tmp.mode == 1 else 'cl_G'
y_cl     = tmp.emulators[cl_key].predict(thetafid).reshape((1, N_Z_BINS, N_Z_BINS, -1))[0]
ell, emm = hp.Alm.getlm(gen_lmax)
params   = {qty: tmp.emulators[qty].predict(thetafid) for qty in MODE_QUANTITIES[tmp.mode] if qty != cl_key}

def get_xlm(xlm_real, xlm_imag):
    ell, emm = hp.Alm.getlm(gen_lmax)
    _xlm_real = torch.zeros(N_Z_BINS, len(ell), dtype=torch.double)
    _xlm_imag = torch.zeros_like(_xlm_real)
    _xlm_real[:,ell > 1] = xlm_real
    _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
    xlm = _xlm_real + 1j * _xlm_imag
    return xlm
    
def generate_xlm():
    xlm_real = torch.tensor(np.random.normal(size=(N_Z_BINS, (ell > 1).sum())), dtype=torch.double)
    xlm_imag = torch.tensor(np.random.normal(size=(N_Z_BINS, ((ell > 1) & (emm > 0)).sum())), dtype=torch.double)
    xlm = get_xlm(xlm_real, xlm_imag)
    return xlm, xlm_real, xlm_imag

def get_g_obs(g1,g2,sigma):
    g1_obs = np.zeros_like(g1)
    g2_obs = np.zeros_like(g1)
    for i in range(N_Z_BINS):
        g1_obs[i] = np.random.normal(g1[i],sigma[i]) 
        g2_obs[i] = np.random.normal(g2[i],sigma[i]) 
    return torch.tensor(g1_obs), torch.tensor(g2_obs)

def get_kappa_KS(g1_obs, g2_obs):
    k_KS_list = []
    for i in range(N_Z_BINS):
        k_KS_i = trf.shear2conv(g1_obs[i], g2_obs[i])    
        k_KS_list.append(k_KS_i.numpy())
    return np.array(k_KS_list)

xlm, xlm_real, xlm_imag = generate_xlm()
ylm           = tmp.apply_cl(xlm, y_cl)
g1_list = []
g2_list = []
kappa_list = []
m = config.mb_init

for j in range(N_Z_BINS):
    x = trf.Alm2Map.apply(ylm[j], nside, gen_lmax)
    k = tmp.compute_k(x, j, params)
    g1, g2 = trf.conv2shear(torch.tensor(k), lmax, tmp.pixwin_ell_filter)
    g1 = (1 + m[j]) * g1.numpy() 
    g2 = (1 + m[j]) * g2.numpy() 
    k  = k.numpy()
    kappa_list.append(k)
    g1_list.append(g1)
    g2_list.append(g2)
k             = np.array(kappa_list)
g1            = np.array(g1_list)
g2            = np.array(g2_list)
g1_obs,g2_obs = get_g_obs(g1,g2,sigma)


def save_datafile(N, g1_obs, g2_obs, mask, xlm_real, xlm_imag, kappa, theta, mb, outpath=config.datafile):
    with h5.File(outpath, 'w') as hf:
        hf['N']        = N
        hf['g1_obs']   = g1_obs
        hf['g2_obs']   = g2_obs
        hf['xlm_real'] = xlm_real.numpy()
        hf['xlm_imag'] = xlm_imag.numpy()
        hf['kappa']    = kappa
        hf['mask']     = mask
        hf['theta']    = theta
        hf['mb_true']  = mb



print('Saving...')
save_datafile(N, g1_obs, g2_obs, mask, xlm_real, xlm_imag, k, config.thetafid, config.mb_init)
