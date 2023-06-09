import sys
sys.path.append('/home/ssarmabo/karmic_harmonies/healcorr')
import numpy as np
import healpy as hp
import healcorr

def get_theta_bins(nside, theta_max=300., nbins=15):
    arcmin2rad = (1. / 60.) * (np.pi/ 180.)
    pix_size     = hp.pixelfunc.nside2resol(nside, arcmin=True)
    theta_bins = np.logspace(np.log10(pix_size), np.log10(theta_max), nbins + 1) * arcmin2rad
    theta_bin_centre = np.sqrt(theta_bins[1:] * theta_bins[:-1])
    return theta_bins, theta_bin_centre

def get_corrfunc(kappa_maps, mask, bins):
    corr = healcorr.compute_corr(kappa_maps, mask=mask.astype(bool), bins=bins, premasked=False, cross_correlate=True, verbose=True)
    return corr    

def get_kappa_bins_1ptfunc(KAPPA_STD_TRUE, nbins, n_kappabins=26):
    kappa_bins = []
    for i in range(nbins):
        kappa_bins_i = np.linspace(-1.5 * KAPPA_STD_TRUE[i], 3. * KAPPA_STD_TRUE[i], n_kappabins)
        kappa_bins.append(kappa_bins_i)

    return np.array(kappa_bins)    

def compute_1ptfunc(kappa_maps, kappa_bins):
    nbins = kappa_maps.shape[0]
    pdf = []
    for i in range(nbins):
        pdf_i, _ = np.histogram(kappa_maps[i], kappa_bins[i], density=True)
        pdf.append(pdf_i)
    return np.array(pdf)