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

    alm_real = a_lm.real * ell_filter
    alm_imag = a_lm.imag * ell_filter

    a_lm_new = alm_real + 1j * alm_imag
    return hp.sphtfunc.alm2map(a_lm_new, nside)    

def get_theta0(kappa_map):
    σ2     = np.std(kappa_map)
    shift0 = -2. * kappa_map.min()
    σg2    = np.log(1. + σ2 / shift0**2)
    return np.array([shift0, σg2])

def logP_LN(theta, x):
    λ, σ2 = theta
    if (λ < 0.) or (σ2 < 0.):
        return np.inf
    z = np.log(x + λ)
    μ = np.log(λ) - 0.5 * σ2
    logL = np.sum(-0.5 * (z - μ)**2 / σ2 - 0.5 * np.log(σ2) - z)
    if np.isnan(logL):
        return np.inf
    return -logL

def fit_lognormal_pars(kappa_maps):
    nbins = kappa_maps.shape[0]    
    
    from scipy.optimize import minimize

    x_opt_list = []
    for i in range(nbins):
        print("i: %d"%(i))
        theta0 = get_theta0(kappa_maps[i])
        results = minimize(logP_LN, theta0, args=(kappa_maps[i]), method='Powell')
        print("Optimal parameters: "+str(results['x']))
        x_opt_list.append(results['x'])

    return np.array(x_opt_list)    