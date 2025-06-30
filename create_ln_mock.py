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

nside    = config.analysis['nside']
nbins    = config.analysis['nbins']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside 
N_Z_BINS = config.analysis['nbins']
pixwin   = config.analysis['pixwin']
mask         = hp.fitsfunc.read_map(config.maskfile)
boolean_mask = mask.astype(bool)
#=== For shape noise ================
sigma_e  = config.analysis['sigma_e']
N        = np.load(config.N_map)
sigma    = sigma_e / np.sqrt(N + 1e-25)
#=====================================
thetafid = torch.tensor(config.thetafid,dtype=torch.double)
#============================================================
print("Initializing sampler....")
tmp = np.zeros((nbins,hp.nside2npix(nside)))
tmp = KarmmaSampler(tmp, tmp, tmp, tmp, lmax, gen_lmax,pixwin=pixwin,shift_file=config.shift_file,mean_g_file=config.mean_g_file,ycl_file=config.y_cl_file,thetafid=config.thetafid)
print("Done initializing sampler....")

ell, emm = hp.Alm.getlm(gen_lmax)
ycl      = tmp.cl_emu.predict(thetafid).reshape((1, N_Z_BINS, N_Z_BINS, -1))[0].numpy()
mean_g   = tmp.mean_g_emu.predict(thetafid)[0].numpy()
shift    = tmp.shift_emu.predict(thetafid)[0].numpy()

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
    return xlm, [xlm_real,xlm_imag]

def generate_mock_y_lm():
    xlm,_xlm = generate_xlm()
    return apply_cl(xlm, ycl), _xlm

def get_y_maps():
    y_lm,xlm = generate_mock_y_lm()
    y_maps = []
    for i in range(nbins):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)    
    return np.array(y_maps),xlm    

def low_pass_filter(map,nside):
    map_lm = hp.map2alm(map,lmax=3*nside-1)
    ell,emm = hp.Alm.getlm(3*nside-1)
    map_lm[ell>2*nside]=0.+0.*1j
    return hp.alm2map(map_lm,nside=nside)

def get_LN_shear(y_maps):
    g1_list = []
    g2_list = []
    k_list = []
    for i in range(nbins):
        k_nf = np.exp(y_maps[i] + mean_g[i]) - shift[i]
        #k = low_pass_filter(k_nf, nside)
        k = k_nf
        k_list.append(k)
        g1, g2 = trf.conv2shear(torch.tensor(k), lmax,tmp.pixwin_ell_filter)
        g1 = g1.numpy() * mask
        g2 = g2.numpy() * mask
        g1_list.append(g1)
        g2_list.append(g2)    

    g1 = np.array(g1_list)
    g2 = np.array(g2_list)    
    k_arr  = np.array(k_list)
    
    return g1, g2, k_arr  
def get_g_obs(g1,g2,sigma):
    g1_obs = np.zeros_like(g1)
    g2_obs = np.zeros_like(g1)
    for i in range(4):
        g1_obs[i] = np.random.normal(g1[i],sigma[i]) * mask
        g2_obs[i] = np.random.normal(g2[i],sigma[i]) * mask
    return g1_obs,g2_obs

def save_datafile(N,g1_obs,g2_obs,mask,xlm,kappa,theta,outpath=config.datafile):
    hf = h5.File(outpath, 'w')
    hf.create_dataset('N',       data   = N)
    hf.create_dataset('g1_obs',  data   = g1_obs)
    hf.create_dataset('g2_obs',  data   = g2_obs)
    hf.create_dataset('xlm_real',data   = xlm[0])
    hf.create_dataset('xlm_imag',data   = xlm[1])
    hf.create_dataset('kappa',   data   = kappa)
    hf.create_dataset('mask',    data   = mask)
    hf.create_dataset('theta',   data   = theta)
    hf.close()

y_maps,xlm        = get_y_maps()
g1, g2, k_arr     = get_LN_shear(y_maps)
g1_obs,g2_obs     = get_g_obs(g1,g2,sigma)

print('Saving...')
save_datafile(N,g1_obs,g2_obs,mask,xlm,k_arr,config.thetafid)
