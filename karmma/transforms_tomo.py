import math
import numpy as np
import healpy as hp
import torch
import multiprocessing as mp

# ========================================================================
# Worker functions for multiprocessing (must be at module level)
# ========================================================================

def mp_map2alm_worker(args):
    """Worker function for Map2Alm multiprocessing"""
    i, m_data, lmax = args
    alm_result = hp.map2alm(m_data, lmax=lmax, use_pixel_weights=True)
    return i, alm_result

def mp_alm2map_worker(args):
    """Worker function for Alm2Map multiprocessing"""
    i, alm_data, nside, lmax = args
    m_result = hp.alm2map(alm_data, nside, lmax=lmax)
    return i, m_result

def mp_alm2map_backward_worker(args):
    """Worker function for Alm2Map backward pass"""
    i, grad_data, lmax, nside = args
    grad_out_alm = hp.map2alm(grad_data, lmax=lmax, use_pixel_weights=True)
    
    # Apply scaling factor
    _, emm = hp.Alm.getlm(lmax)
    a = np.ones(len(emm))
    a[emm > 0] = 2
    grad_alm = a * hp.nside2npix(nside) / (4 * math.pi) * grad_out_alm
    return i, grad_alm

def mp_map2alm_backward_worker(args):
    """Worker function for Map2Alm backward pass"""
    i, grad_alm_data, nside, lmax = args
    
    # Apply scaling factor
    _, emm = hp.Alm.getlm(lmax)
    a = np.ones(len(grad_alm_data))
    a[emm > 0] = 0.5
    scaled_grad = a * grad_alm_data
    
    grad_out_m = hp.alm2map(scaled_grad, nside, lmax=lmax)
    grad_m = 4 * math.pi / hp.nside2npix(nside) * grad_out_m
    return i, grad_m

def mp_alm2map_spin_worker(args):
    """Worker function for Alm2MapSpin multiprocessing"""
    i, elm_data, blm_data, nside, lmax = args
    inputs = [np.zeros_like(elm_data), elm_data, blm_data]
    _, q, u = hp.alm2map(inputs, nside, lmax=lmax)
    return i, q, u

def mp_map2alm_spin_worker(args):
    """Worker function for Map2AlmSpin multiprocessing"""
    i, q_data, u_data, lmax = args
    inputs = [np.zeros_like(q_data), q_data, u_data]
    _, elm, blm = hp.map2alm(inputs, lmax=lmax, use_pixel_weights=True)
    return i, elm, blm

def mp_alm2map_spin_backward_worker(args):
    """Worker function for Alm2MapSpin backward pass"""
    i, q_grad_data, u_grad_data, lmax, nside = args
    inputs = [np.zeros_like(q_grad_data), q_grad_data, u_grad_data]
    _, elm_grad, blm_grad = hp.map2alm(inputs, lmax=lmax, use_pixel_weights=True)
    
    # Apply scaling factor
    _, emm = hp.Alm.getlm(lmax)
    a = np.ones(len(emm))
    a[emm > 0] = 2
    
    elm_grad = a * hp.nside2npix(nside) / (4 * math.pi) * elm_grad
    blm_grad = a * hp.nside2npix(nside) / (4 * math.pi) * blm_grad
    return i, elm_grad, blm_grad

def mp_map2alm_spin_backward_worker(args):
    """Worker function for Map2AlmSpin backward pass"""
    i, elm_grad_data, blm_grad_data, nside, lmax = args
    
    # Apply scaling factor
    _, emm = hp.Alm.getlm(lmax)
    a = np.ones(len(elm_grad_data))
    a[emm > 0] = 0.5
    
    inputs = [np.zeros_like(elm_grad_data), a * elm_grad_data, a * blm_grad_data]
    _, q_grad, u_grad = hp.alm2map(inputs, nside, lmax=lmax)
    
    q_grad = 4 * math.pi / hp.nside2npix(nside) * q_grad
    u_grad = 4 * math.pi / hp.nside2npix(nside) * u_grad
    return i, q_grad, u_grad

def mp_udgrade_worker(args):
    """Worker function for UDGrade multiprocessing"""
    i, m_data, out_nside = args
    ud_m = hp.ud_grade(m_data, out_nside)
    return i, ud_m

def mp_udgrade_backward_worker(args):
    """Worker function for UDGrade backward pass"""
    i, grad_data, in_nside, out_nside = args
    if out_nside > in_nside:
        fac = (out_nside / in_nside) ** 2.
    else:
        fac = (in_nside / out_nside) ** (-2.)
    
    grad = hp.ud_grade(grad_data, in_nside) * fac
    return i, grad

# ========================================================================
# Tomographic Multiprocessing Classes
# ========================================================================

class Alm2MapTomoMP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alms_tomo, nside, lmax, max_workers=4):
        """alms_tomo shape: (n_tomo, n_alms)"""
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.n_tomo = alms_tomo.shape[0]
        ctx.max_workers = max_workers
        
        # Convert to numpy and prepare arguments
        args_list = [(i, alms_tomo[i].numpy(), nside, lmax) 
                     for i in range(ctx.n_tomo)]
        
        # Multiprocessing approach
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_alm2map_worker, args_list)
        
        # Sort results by index and stack into tensor
        results.sort(key=lambda x: x[0])
        m_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        return m_tomo
    
    @staticmethod
    def backward(ctx, grad_output):
        nside = ctx.nside
        lmax = ctx.lmax
        n_tomo = ctx.n_tomo
        max_workers = ctx.max_workers
        
        # Prepare arguments for parallel backward
        args_list = [(i, grad_output[i].numpy(), lmax, nside) for i in range(n_tomo)]
        
        # Multiprocessing in backward pass
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_alm2map_backward_worker, args_list)
        
        # Sort and stack results
        results.sort(key=lambda x: x[0])
        grad_alms_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        return grad_alms_tomo, None, None, None

class Map2AlmTomoMP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m_tomo, lmax, max_workers=4):
        """m_tomo shape: (n_tomo, n_pix)"""
        ctx.nside = hp.npix2nside(m_tomo.shape[1])
        ctx.lmax = lmax
        ctx.n_tomo = m_tomo.shape[0]
        ctx.max_workers = max_workers
        
        # Convert to numpy and prepare arguments
        args_list = [(i, m_tomo[i].numpy(), lmax) 
                     for i in range(ctx.n_tomo)]
        
        # Multiprocessing approach
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_map2alm_worker, args_list)
        
        # Sort results by index and stack into tensor
        results.sort(key=lambda x: x[0])
        alm_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        return alm_tomo
    
    @staticmethod
    def backward(ctx, grad_output):
        nside = ctx.nside
        lmax = ctx.lmax
        n_tomo = ctx.n_tomo
        max_workers = ctx.max_workers
        
        # Prepare arguments for parallel backward
        args_list = [(i, grad_output[i].numpy(), nside, lmax) for i in range(n_tomo)]
        
        # Multiprocessing in backward pass
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_map2alm_backward_worker, args_list)
        
        # Sort and stack results
        results.sort(key=lambda x: x[0])
        grad_m_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        return grad_m_tomo, None, None

class Alm2MapSpinTomoMP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, elm_tomo, blm_tomo, nside, lmax, max_workers=4):
        """elm_tomo, blm_tomo shape: (n_tomo, n_alms)"""
        ctx.nside = nside
        ctx.lmax = lmax
        ctx.n_tomo = elm_tomo.shape[0]
        ctx.max_workers = max_workers
        
        # Convert to numpy and prepare arguments
        args_list = [(i, elm_tomo[i].numpy(), blm_tomo[i].numpy(), nside, lmax) 
                     for i in range(ctx.n_tomo)]
        
        # Multiprocessing approach
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_alm2map_spin_worker, args_list)
        
        # Sort results by index and stack into tensors
        results.sort(key=lambda x: x[0])
        q_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        u_tomo = torch.stack([torch.tensor(result[2]) for result in results])
        return q_tomo, u_tomo
    
    @staticmethod
    def backward(ctx, q_grad_tomo, u_grad_tomo):
        nside = ctx.nside
        lmax = ctx.lmax
        n_tomo = ctx.n_tomo
        max_workers = ctx.max_workers
        
        # Prepare arguments for parallel backward
        args_list = [(i, q_grad_tomo[i].numpy(), u_grad_tomo[i].numpy(), lmax, nside) 
                     for i in range(n_tomo)]
        
        # Multiprocessing in backward pass
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_alm2map_spin_backward_worker, args_list)
        
        # Sort and stack results
        results.sort(key=lambda x: x[0])
        elm_grad_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        blm_grad_tomo = torch.stack([torch.tensor(result[2]) for result in results])
        return elm_grad_tomo, blm_grad_tomo, None, None, None

class Map2AlmSpinTomoMP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_tomo, u_tomo, lmax, max_workers=4):
        """q_tomo, u_tomo shape: (n_tomo, n_pix)"""
        ctx.nside = hp.npix2nside(q_tomo.shape[1])
        ctx.lmax = lmax
        ctx.n_tomo = q_tomo.shape[0]
        ctx.max_workers = max_workers
        
        # Convert to numpy and prepare arguments
        args_list = [(i, q_tomo[i].numpy(), u_tomo[i].numpy(), lmax) 
                     for i in range(ctx.n_tomo)]
        
        # Multiprocessing approach
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_map2alm_spin_worker, args_list)
        
        # Sort results by index and stack into tensors
        results.sort(key=lambda x: x[0])
        elm_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        blm_tomo = torch.stack([torch.tensor(result[2]) for result in results])
        return elm_tomo, blm_tomo
    
    @staticmethod
    def backward(ctx, elm_grad_tomo, blm_grad_tomo):
        nside = ctx.nside
        lmax = ctx.lmax
        n_tomo = ctx.n_tomo
        max_workers = ctx.max_workers
        
        # Prepare arguments for parallel backward
        args_list = [(i, elm_grad_tomo[i].numpy(), blm_grad_tomo[i].numpy(), nside, lmax) 
                     for i in range(n_tomo)]
        
        # Multiprocessing in backward pass
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_map2alm_spin_backward_worker, args_list)
        
        # Sort and stack results
        results.sort(key=lambda x: x[0])
        q_grad_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        u_grad_tomo = torch.stack([torch.tensor(result[2]) for result in results])
        return q_grad_tomo, u_grad_tomo, None, None

class UDGradeTomoMP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m_tomo, out_nside, max_workers=4):
        """m_tomo shape: (n_tomo, n_pix)"""
        ctx.in_nside = hp.npix2nside(m_tomo.shape[1])
        ctx.out_nside = out_nside
        ctx.n_tomo = m_tomo.shape[0]
        ctx.max_workers = max_workers
        
        # Convert to numpy and prepare arguments
        args_list = [(i, m_tomo[i].numpy(), out_nside) 
                     for i in range(ctx.n_tomo)]
        
        # Multiprocessing approach
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_udgrade_worker, args_list)
        
        # Sort results by index and stack into tensor
        results.sort(key=lambda x: x[0])
        ud_m_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        return ud_m_tomo
    
    @staticmethod
    def backward(ctx, grad_output):
        in_nside = ctx.in_nside
        out_nside = ctx.out_nside
        n_tomo = ctx.n_tomo
        max_workers = ctx.max_workers
        
        # Prepare arguments for parallel backward
        args_list = [(i, grad_output[i].numpy(), in_nside, out_nside) for i in range(n_tomo)]
        
        # Multiprocessing in backward pass
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(mp_udgrade_backward_worker, args_list)
        
        # Sort and stack results
        results.sort(key=lambda x: x[0])
        grad_tomo = torch.stack([torch.tensor(result[1]) for result in results])
        return grad_tomo, None, None

# ========================================================================
# Tomographic Utility Functions
# ========================================================================

def shear2conv_tomo(g1_tomo, g2_tomo, lmax=None, max_workers=4):
    """
    Convert tomographic shear fields to convergence fields
    g1_tomo, g2_tomo shape: (n_tomo, n_pix)
    Returns: k_tomo shape: (n_tomo, n_pix)
    """
    nside = hp.npix2nside(g1_tomo.shape[1])
    
    # Get spherical harmonic coefficients for shear
    gelm_tomo, _ = Map2AlmSpinTomoMP.apply(g1_tomo, g2_tomo, lmax, max_workers)
    
    # Apply conversion from shear to convergence
    lmax_actual = hp.Alm.getlmax(gelm_tomo.shape[1])
    l, m = hp.Alm.getlm(lmax_actual)
    l = torch.tensor(l, dtype=torch.double)
    
    good_ls = l > 1
    fac = torch.zeros_like(l)
    l_good = l[good_ls]
    fac[good_ls] = -torch.sqrt(l_good * (l_good + 1) / ((l_good + 2) * (l_good - 1)))
    
    # Apply factor to all tomographic bins
    kelm_tomo = fac.unsqueeze(0) * gelm_tomo  # Broadcasting over tomo dimension
    
    # Convert back to maps
    k_tomo = Alm2MapTomoMP.apply(kelm_tomo, nside, lmax_actual, max_workers)
    
    return k_tomo

def conv2shear_tomo(k_tomo, lmax=None, pixwin=None, max_workers=4):
    """
    Convert tomographic convergence fields to shear fields
    k_tomo shape: (n_tomo, n_pix)
    Returns: g1_tomo, g2_tomo shapes: (n_tomo, n_pix)
    """
    nside = hp.npix2nside(k_tomo.shape[1])
    
    # Get spherical harmonic coefficients for convergence
    kelm_tomo = Map2AlmTomoMP.apply(k_tomo, lmax, max_workers)
    
    # Apply pixel window if provided
    if pixwin is not None:
        if pixwin.dim() == 1:
            # Broadcast pixel window over tomographic dimension
            kelm_tomo = kelm_tomo * pixwin.unsqueeze(0)
        else:
            kelm_tomo = kelm_tomo * pixwin
    
    # Apply conversion from convergence to shear
    lmax_actual = hp.Alm.getlmax(kelm_tomo.shape[1])
    l, m = hp.Alm.getlm(lmax_actual)
    l = torch.tensor(l, dtype=torch.double)
    
    good_ls = l > 0
    fac = torch.zeros_like(l)
    l_good = l[good_ls]
    fac[good_ls] = -torch.sqrt((l_good + 2) * (l_good - 1) / (l_good * (l_good + 1)))
    
    # Apply factor to all tomographic bins
    gelm_tomo = fac.unsqueeze(0) * kelm_tomo  # Broadcasting over tomo dimension
    gblm_tomo = torch.zeros_like(kelm_tomo)
    
    # Convert back to maps
    g1_tomo, g2_tomo = Alm2MapSpinTomoMP.apply(gelm_tomo, gblm_tomo, nside, lmax_actual, max_workers)
    
    return g1_tomo, g2_tomo

