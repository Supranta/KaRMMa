import numpy as np
import matplotlib.pyplot as plt
import skymapper as skm

def plot_nz(nz):
    nbins = nz.shape[0]

    plt.xlabel('$z$')
    plt.ylabel('$n(z)$')
    for i in range(nbins):
        plt.xlim(0., 2.)
        plt.plot(nz[i,:,0], nz[i,:,1], label='Bin %d'%(i+1))
    plt.legend()
    plt.show()    
    
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

def plot_map_skm(ax, kappa_map, mask, text, proj_data, minmax=None, cmap='viridis', sep=15):
    proj, ra, dec = proj_data
    boolean_mask = mask.astype(bool)
    if minmax is None:
        vmin, vmax = np.percentile(kappa_map[boolean_mask], [10, 90])
    else:
        vmin, vmax = minmax
    map = skm.Map(proj, ax=ax)
    map.grid(sep=sep, parallel_fmt=lambda x: '', meridian_fmt=lambda x: '')
    mappable = map.healpix(kappa_map * mask, vmin=vmin, vmax=vmax, cmap=cmap)  
    cb = map.colorbar(mappable, cb_label="$\\kappa$")
    map.text(340, 10, text, 0)
    map.focus(ra, dec)