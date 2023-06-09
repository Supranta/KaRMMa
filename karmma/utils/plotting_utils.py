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
    
def plot_corr_cl(nbins, sample_summaries, true_summaries, bincentre, savename=None):
    corr_arr, pseudocl_arr   = sample_summaries
    corr_true, pseudocl_true = true_summaries
    theta_bincentre, ell_bincentre = bincentre
    
    corr_mean        = corr_arr.mean(0)
    corr_lo, corr_hi = np.percentile(corr_arr, [2.5, 97.5], axis=0)

    pseudocl_mean            = pseudocl_arr.mean(0)
    pseudocl_lo, pseudocl_hi = np.percentile(pseudocl_arr, [2.5, 97.5], axis=0)
    
    fig, ax = plt.subplots(nbins+1,nbins+1,figsize=(3*nbins,3*nbins))

    for i in range(nbins+1):
        ax[i,i].axis('off')

    for i in range(nbins):
        for j in range(nbins):        
            if not (i <= j-1):
                ax[i+1,j].set_xlabel(r'$\theta$ (arcmin)')
                ax[i+1,j].set_ylabel(r'$\xi_{%d%d}(\theta)$'%(j+1,i+1))
                ax[i+1,j].loglog(theta_bincentre, corr_true[i,j], 'k-', label='True')
                ax[i+1,j].loglog(theta_bincentre, corr_mean[i,j], 'b-', label='Sample')
                ax[i+1,j].fill_between(theta_bincentre, corr_lo[i,j], corr_hi[i,j], color='b', alpha=0.3)
            if not (j <= i-1):
                ax[i,j+1].set_xlabel(r'$\ell$')
                ax[i,j+1].set_ylabel(r'$C_{%d%d}(\ell)$'%(i+1,j+1))
                ax[i,j+1].loglog(ell_bincentre, pseudocl_true[i,j], 'k-', label='True')
                ax[i,j+1].loglog(ell_bincentre, pseudocl_mean[i,j], 'b-', label='Sample')    
                ax[i,j+1].fill_between(ell_bincentre, pseudocl_lo[i,j], pseudocl_hi[i,j], color='b', alpha=0.3)
    ax[1,0].legend()
    
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.show()                
    
def plot_1pt_pdf(nbins, kappa_bincentre, kappapdf_arr, kappa_pdf_true, savename=None):
    kappapdf_mean            = kappapdf_arr.mean(0)
    kappapdf_lo, kappapdf_hi = np.percentile(kappapdf_arr, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(2,nbins,figsize=(12.,6.))

    for i in range(nbins):
        for j in range(2):
            ax[j,i].set_xlabel('$\kappa$')
            ax[j,i].set_ylabel('$P(\kappa)$')
            ax[j,i].plot(kappa_bincentre[i], kappapdf_mean[i],  'b-')    
            ax[j,i].fill_between(kappa_bincentre[i], kappapdf_lo[i], kappapdf_hi[i], color='b', alpha=0.3)
            ax[j,i].plot(kappa_bincentre[i], kappa_pdf_true[i], 'k-')

        ax[1,i].set_yscale('log')       

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.show()    
    
def plot_crosscorr(nbins, theta_bincentre, crosscorr_arr, savename=None):
    crosscorr_mean = crosscorr_arr.mean(0)
    crosscorr_lo, crosscorr_hi = np.percentile(crosscorr_arr, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(1,nbins,figsize=(3.*nbins,3.))

    for i in range(nbins):
        ax[i].set_xlabel(r'$\theta$ (arcmin)')
        ax[i].set_ylabel(r'$\rho_c(\theta)$')
        ax[i].semilogx(theta_bincentre, crosscorr_mean[i], 'b-')
        ax[i].fill_between(theta_bincentre, crosscorr_lo[i], crosscorr_hi[i], color='b', alpha=0.3)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.show()    