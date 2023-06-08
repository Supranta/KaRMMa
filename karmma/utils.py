import numpy as np
import healpy as hp
import pyccl
import matplotlib.pyplot as plt
import skymapper as skm

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

def plot_nz(nz):
    nbins = nz.shape[0]

    plt.xlabel('$z$')
    plt.ylabel('$n(z)$')
    for i in range(nbins):
        plt.xlim(0., 2.)
        plt.plot(nz[i,:,0], nz[i,:,1], label='Bin %d'%(i+1))
    plt.legend()
    plt.show()    
    
def get_filtered_map(hp_map, ell_filter, nside):
    a_lm = hp.sphtfunc.map2alm(hp_map)    

    a_lm[ell_filter].real = 0.
    a_lm[ell_filter].imag = 0.

    return hp.sphtfunc.alm2map(a_lm, nside)    

def getCatalog(size=10000, survey=None):
    # dummy catalog: uniform on sphere
    # Marsaglia (1972)
    xyz = np.random.normal(size=(size, 3))
    r = np.sqrt((xyz**2).sum(axis=1))
    dec = np.arccos(xyz[:,2]/r) / skm.DEG2RAD - 90
    ra = - np.arctan2(xyz[:,0], xyz[:,1]) / skm.DEG2RAD

    if survey is not None:
        inside = survey.contains(ra, dec)
        ra = ra[inside]
        dec = dec[inside]

    return ra, dec

def get_proj_data():
    size = 100000
    des = skm.survey.DES()
    ra, dec = getCatalog(size, survey=des)

    crit = skm.stdDistortion
    proj = skm.Albers.optimize(ra, dec, crit=crit)
    
    return [proj, ra, dec]

def plot_map_skm(ax, kappa_map, mask, text, proj_data, plot_masked=True, cmap='viridis', sep=15):
    proj, ra, dec = proj_data
    boolean_mask = mask.astype(bool)
    vmin, vmax = np.percentile(kappa_map[boolean_mask], [10, 90])
    map = skm.Map(proj, ax=ax)
    map.grid(sep=sep, parallel_fmt=lambda x: '', meridian_fmt=lambda x: '')
    if(plot_masked):
        mappable = map.healpix(kappa_map * mask, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        mappable = map.healpix(kappa_map, vmin=vmin, vmax=vmax, cmap=cmap)
    cb = map.colorbar(mappable, cb_label="$\\kappa$")
    map.text(340, 10, text, 0)
    map.focus(ra, dec)