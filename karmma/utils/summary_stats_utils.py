import sys
sys.path.append('/home/ssarmabo/karmic_harmonies/healcorr')
#sys.path.append('/groups/erozo/ssarmabo/softs/healcorr/')
import numpy as np
import healpy as hp
import pymaster as nmt
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

def get_kappa_bins_1ptfunc(KAPPA_STD_TRUE, nbins, n_kappabins=46):
    kappa_bins = []
    for i in range(nbins):
        kappa_bins_i = np.linspace(-4. * KAPPA_STD_TRUE[i], 5. * KAPPA_STD_TRUE[i], n_kappabins)
        kappa_bins.append(kappa_bins_i)

    return np.array(kappa_bins)    

def get_1ptfunc(kappa_maps, kappa_bins, mask):
    nbins = kappa_maps.shape[0]
    pdf = []
    for i in range(nbins):
        pdf_i, _ = np.histogram(kappa_maps[i][mask.astype(bool)], kappa_bins[i], density=True)
        pdf.append(pdf_i)
    return np.array(pdf)

def get_nmt_ell_bins(nside, n_ell_bins=17):
    ell_bins     = np.ceil(np.logspace(np.log10(3), np.log10(2 * nside), n_ell_bins)).astype(int)[1:]
    nmt_ell_bins = nmt.NmtBin.from_edges(ell_bins[:-1], ell_bins[1:])
    eff_ell_arr  = nmt_ell_bins.get_effective_ells()
    return nmt_ell_bins, ell_bins, eff_ell_arr

def get_pseudo_cls(kappa, mask, nmt_ell_bins):
    nbins = kappa.shape[0]
    nmt_kappa_fields = [nmt.NmtField(mask, [kappa[i]]) for i in range(nbins)]
    N_ell = nmt_ell_bins.get_n_bands()
    cls = np.zeros((nbins, nbins, N_ell))
    for i in range(nbins):
        for j in range(i+1):
            cl_ij = nmt.compute_full_master(nmt_kappa_fields[i], nmt_kappa_fields[j], nmt_ell_bins)
            cls[i,j] = cl_ij
            cls[j,i] = cl_ij
    return cls

def get_cross_corr(kappa1, kappa2, mask, theta_bins):
    nbins = kappa1.shape[0]
    cross_corr_list = []
    for i in range(nbins):
        print("Computing cross-corr for bin: %d"%(i+1))
        kappa_map_corr = np.array([kappa1[i], kappa2[i]])
        corr = healcorr.compute_corr(kappa_map_corr, mask=mask.astype(bool), bins=theta_bins, premasked=False, cross_correlate=True, verbose=False)
        cross_corr = corr[0,1]/ np.sqrt(corr[0,0] * corr[1,1])
        cross_corr_list.append(cross_corr)

    return np.array(cross_corr_list)

def get_neighbor_maps(hp_map):
    npix = hp_map.shape[0]
    nside = hp.npix2nside(npix)
    neighbour_indices = hp.get_all_neighbours(nside, np.arange(npix))
    neighbor_maps = []
    for i in range(8):
        neighbor_maps.append(hp_map[neighbour_indices[i]])
    return np.array(neighbor_maps)

def get_kappa_peaks(hp_map, mask):
    neighbor_maps     = get_neighbor_maps(hp_map)
    max_neighbour_map = np.max(neighbor_maps, axis=0)
    select_peaks      = (hp_map > max_neighbour_map) & mask
    return hp_map[select_peaks]

def get_kappa_troughs(hp_map, mask):
    neighbor_maps     = get_neighbor_maps(hp_map)
    min_neighbour_map = np.min(neighbor_maps, axis=0)
    select_troughs      = (hp_map < min_neighbour_map) & mask
    return hp_map[select_troughs]

def get_counts(kappa_map, kappa_bins, mask, flag='peak'):
    if flag=='peak':
        kappa_features = get_kappa_peaks(kappa_map, mask)
    elif flag=='void':
        kappa_features = get_kappa_troughs(kappa_map, mask)
    counts, _ = np.histogram(kappa_features, kappa_bins, density=True)
    return counts

def get_tomo_counts(kappa_map, kappa_bins, mask, flag='peak'):
    counts_list = []
    nbins = kappa_map.shape[0]
    counts_list_patch = []
    for i in range(nbins):
        counts = get_counts(kappa_map[i], kappa_bins[i], mask.astype(bool), flag)
        counts_list.append(counts)
    return np.array(counts_list)


