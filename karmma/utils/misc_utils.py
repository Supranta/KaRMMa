import numpy as np
import healpy as hp
import pyccl

def get_cl(Omega_c, sigma_8, nz_data, N_Z_BINS, gen_lmax):
    cosmo = pyccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=0.046,
        h=0.7,
        sigma8=sigma_8,
        n_s=0.97
    )
    tracers = []
    for i in range(N_Z_BINS):
        tracer_i = pyccl.tracers.WeakLensingTracer(cosmo, nz_data[i].T)
        tracers.append(tracer_i)

    ell = np.arange(gen_lmax + 1)
    cl = np.zeros((N_Z_BINS, N_Z_BINS, len(ell)))
    for i in range(N_Z_BINS):
        for j in range(i+1):
            cl_ij = pyccl.cls.angular_cl(cosmo, tracers[i], tracers[j], ell)
            cl[i,j] = cl_ij
            cl[j,i] = cl_ij

    cl[:,:,0] = 1e-21 * np.eye(N_Z_BINS)
    cl[:,:,1] = 1e-21 * np.eye(N_Z_BINS)
    
    return cl
    
def get_filtered_map(hp_map, ell_filter, nside):
    a_lm = hp.sphtfunc.map2alm(hp_map)    

    a_lm[ell_filter].real = 0.
    a_lm[ell_filter].imag = 0.

    return hp.sphtfunc.alm2map(a_lm, nside)    