import numpy as np 
from scipy.optimize import minimize
import torch
from sklearn.decomposition import PCA

class ScalarEmulator:
    def __init__(self, training_data):
        self.cosmology_training = training_data[0]
        self.N_cosmo_params     = self.cosmology_training.shape[1]
        self.N_td               = self.cosmology_training.shape[0]
        self.N                  = self.N_cosmo_params + 1
        self.N_params           = int(0.5*self.N*(self.N+1))
        self.cosmology_max      = self.cosmology_training.max(axis=0)
        self.cosmology_min      = self.cosmology_training.min(axis=0)
        self.cosmology_mid      = 0.5 * (self.cosmology_max + self.cosmology_min)
        scalar                  = training_data[1]
        self.scalar_mean        = torch.Tensor(np.mean(scalar, axis=0)[np.newaxis]).to(torch.double)
        self.scalar_std         = torch.Tensor(np.std(scalar, axis=0)[np.newaxis]).to(torch.double)
        self.scalar_norm        = (torch.Tensor(scalar).to(torch.double) - self.scalar_mean) / self.scalar_std

    def model(self,params,cosmo):
        zeroth  = params[0]
        lin     = params[1:self.N]
        quad    = self.symm_matrix(params[self.N:], self.N_cosmo_params)
        Delta = cosmo - self.cosmology_mid
        linear_terms = Delta @ lin
        quad_terms   = np.sum((Delta @ quad) * Delta, axis=1)
        return zeroth + linear_terms + quad_terms

    def fit_params(self):
        Delta = self.cosmology_training - self.cosmology_mid        
        n = self.N_cosmo_params
        iu, ju = np.triu_indices(n)
        quad_cols = Delta[:, iu] * Delta[:, ju]
        quad_cols[:, iu != ju] *= 2.0
        X = np.concatenate([np.ones((Delta.shape[0], 1)), Delta, quad_cols], axis=1)
        y = self.scalar_norm.numpy().ravel()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.zeroth = beta[0]
        self.lin    = torch.tensor(beta[1:self.N], dtype=torch.double)
        self.quad   = torch.tensor(
            self.symm_matrix(beta[self.N:], self.N_cosmo_params),
            dtype=torch.double
        )
          
    def model_torch(self,cosmo):       
        Delta = cosmo - torch.tensor(self.cosmology_mid, dtype=torch.double)
        return self.zeroth + (self.lin @ Delta) + (Delta.T @ self.quad @ Delta)

    def predict(self, cosmo):
        scalar_norm_pred = self.model_torch(cosmo)      
        scalar_pred      = self.scalar_mean + self.scalar_std * scalar_norm_pred
        return scalar_pred

    def symm_matrix(self,values, n):
        matrix = np.zeros((n, n))
        idx    = 0
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = values[idx]
                idx += 1
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))      
        return matrix
    
class ParamsEmu:
    def __init__(self, training_data):
        self.N_bins = training_data[1].shape[1]
        self.emu    = [ScalarEmulator([training_data[0], training_data[1][:,i]]) for i in range(self.N_bins)]

    def fit_params(self):
        for i in range(self.N_bins):
            self.emu[i].fit_params()
        self.trained = True
    
    def predict(self,cosmo):
        prediction = torch.zeros((1,self.N_bins),dtype=torch.double)
        for i in range(self.N_bins):
            prediction[0,i] = self.emu[i].predict(cosmo)
        return prediction

class ClEmu:
    def __init__(self,training_data,N_PCA=4):
        self.N_PCA          = N_PCA
        cl                  = training_data[1]
        log_cl              = np.log(cl.reshape((cl.shape[0], -1)) + 1e-25)
        #==== all pca quantities we require
        pca                 = PCA(self.N_PCA)
        pca.fit(log_cl)
        pca_coef_training   = pca.transform(log_cl)
        self.PCA_MEAN       = torch.Tensor(np.mean(pca_coef_training, axis=0)[np.newaxis]).to(torch.double)
        self.PCA_STD        = torch.Tensor(np.std(pca_coef_training, axis=0)[np.newaxis]).to(torch.double)
        pca_coeff_norm      = (torch.Tensor(pca_coef_training).to(torch.double) - self.PCA_MEAN) / self.PCA_STD
        self.pca_mean       = torch.Tensor(pca.mean_).to(torch.double)
        self.pca_components = torch.Tensor(pca.components_[:,np.newaxis])
        #=================================
        self.emu = [ScalarEmulator([training_data[0], pca_coeff_norm[:,i].numpy()]) for i in range(self.N_PCA)]
   
    def fit_params(self):
        for i in range(self.N_PCA):
            self.emu[i].fit_params()
        self.trained = True
    
    def predict(self,cosmo):
        log_cl_pred = self.pca_mean
        for i in range(self.N_PCA):
            pca_coeff_i = self.emu[i].predict(cosmo)
            pca_pred_i  = self.PCA_MEAN[:,i] + self.PCA_STD[:,i] * pca_coeff_i
            log_cl_pred = log_cl_pred + pca_pred_i * self.pca_components[i]
        return torch.exp(log_cl_pred)

class ClEmu_Cholesky:
    def __init__(self, training_data, N_PCA=4):
        self.N_PCA = N_PCA
        cl         = training_data[1]  # (N_td, N_bins, N_bins, N_ell)
        N_td, N_bins, _, N_ell = cl.shape
        self.N_bins  = N_bins
        self.N_ell   = N_ell

        # Compute Cholesky at each ell for each training sample
        L = np.linalg.cholesky(cl.transpose(0, 3, 1, 2))  # (N_td, N_ell, N_bins, N_bins)

        # Log-transform the diagonal for unconstrained emulation
        diag_idx = np.arange(N_bins)
        L[:, :, diag_idx, diag_idx] = np.log(L[:, :, diag_idx, diag_idx])

        # Flatten lower triangle only
        self.tril_idx = np.tril_indices(N_bins)
        L_flat = L[:, :, self.tril_idx[0], self.tril_idx[1]]  # (N_td, N_ell, N_bins*(N_bins+1)/2)
        L_flat = L_flat.reshape(N_td, -1)                      # (N_td, N_ell * N_bins*(N_bins+1)/2)

        # PCA
        pca               = PCA(self.N_PCA)
        pca.fit(L_flat)
        pca_coef_training = pca.transform(L_flat)
        self.PCA_MEAN     = torch.Tensor(np.mean(pca_coef_training, axis=0)[np.newaxis]).to(torch.double)
        self.PCA_STD      = torch.Tensor(np.std(pca_coef_training, axis=0)[np.newaxis]).to(torch.double)
        pca_coeff_norm    = (torch.Tensor(pca_coef_training).to(torch.double) - self.PCA_MEAN) / self.PCA_STD
        self.pca_mean     = torch.Tensor(pca.mean_).to(torch.double)
        self.pca_components = torch.Tensor(pca.components_[:, np.newaxis])

        self.emu = [ScalarEmulator([training_data[0], pca_coeff_norm[:, i].numpy()]) for i in range(self.N_PCA)]

    def fit_params(self):
        for i in range(self.N_PCA):
            self.emu[i].fit_params()
        self.trained = True

    def predict(self, cosmo):
        # Reconstruct L_flat from emulator
        L_flat_pred = self.pca_mean.clone()
        for i in range(self.N_PCA):
            pca_coeff_i = self.emu[i].predict(cosmo)
            pca_pred_i  = self.PCA_MEAN[:, i] + self.PCA_STD[:, i] * pca_coeff_i
            L_flat_pred = L_flat_pred + pca_pred_i * self.pca_components[i]

        # Reshape to (N_ell, N_bins*(N_bins+1)/2)
        L_flat_pred = L_flat_pred.reshape(self.N_ell, -1)

        # Reconstruct full L matrix
        diag_idx = torch.arange(self.N_bins)
        L_pred   = torch.zeros(self.N_ell, self.N_bins, self.N_bins, dtype=torch.double)
        L_pred[:, self.tril_idx[0], self.tril_idx[1]] = L_flat_pred

        # Exponentiate diagonal back
        L_pred[:, diag_idx, diag_idx] = torch.exp(L_pred[:, diag_idx, diag_idx])

        # Cl = L @ L.T
        Cl_pred = torch.bmm(L_pred, L_pred.transpose(1, 2))   # (N_ell, N_bins, N_bins)
        return Cl_pred.permute(1, 2, 0).unsqueeze(0)           # (1, N_bins, N_bins, N_ell)
    
class MaskedDerivativeEmu:
    """
    PCA emulator that excludes structurally-zero entries from the regression
    and reinserts them as exact zeros at predict time. Differentiable w.r.t.
    cosmo via index_put.
    """
    def __init__(self, training_data, mask, log_t=False, N_PCA=4):
        self.N_PCA = N_PCA
        self.log_t = log_t
        cl = training_data[1]
        self.trailing_shape = cl.shape[1:]
        self.mask_flat = mask.flatten()                    # numpy bool, shape (prod(trailing),)

        flat_all  = cl.reshape((cl.shape[0], -1))
        flat_kept = flat_all[:, self.mask_flat]

        if log_t:
            flat_kept = np.log(flat_kept + 1e-25)

        from sklearn.decomposition import PCA
        pca = PCA(self.N_PCA)
        pca.fit(flat_kept)
        pca_coef_training = pca.transform(flat_kept)
        self.PCA_MEAN       = torch.Tensor(np.mean(pca_coef_training, axis=0)[np.newaxis]).to(torch.double)
        self.PCA_STD        = torch.Tensor(np.std (pca_coef_training, axis=0)[np.newaxis]).to(torch.double)
        pca_coeff_norm      = (torch.Tensor(pca_coef_training).to(torch.double) - self.PCA_MEAN) / self.PCA_STD
        self.pca_mean       = torch.Tensor(pca.mean_).to(torch.double)
        self.pca_components = torch.Tensor(pca.components_[:, np.newaxis])

        # Pre-build a torch tensor of the mask for use in predict
        self.mask_t = torch.tensor(self.mask_flat, dtype=torch.bool)
        # And the indices where the mask is True — these are stable across calls
        self.kept_idx = torch.tensor(np.where(self.mask_flat)[0], dtype=torch.long)

        self.emu = [ScalarEmulator([training_data[0], pca_coeff_norm[:, i].numpy()])
                    for i in range(self.N_PCA)]

    def fit_params(self):
        for i in range(self.N_PCA):
            self.emu[i].fit_params()
        self.trained = True

    def predict(self, cosmo):
        flat_pred = self.pca_mean
        for i in range(self.N_PCA):
            coeff_i = self.emu[i].predict(cosmo)
            pred_i  = self.PCA_MEAN[:, i] + self.PCA_STD[:, i] * coeff_i
            flat_pred = flat_pred + pred_i * self.pca_components[i]

        if self.log_t:
            flat_pred = torch.exp(flat_pred)

        # flat_pred has length N_kept (shape can be (1, N_kept) due to broadcasting).
        # Build a length-N_full vector with zeros at masked positions.
        flat_pred = flat_pred.reshape(-1)                       # (N_kept,)
        full_flat = torch.zeros(self.mask_t.shape[0], dtype=torch.double)
        full_flat = full_flat.index_put((self.kept_idx,), flat_pred)
        return full_flat.reshape(self.trailing_shape)


class L_emu:
    """
    Cosmology emulator for the L (Cholesky factor) Taylor coefficients.
    Differentiable w.r.t. cosmo and Δz through torch autograd.

    Usage:
        l_emu = L_emu('/path/to/stacked.npz')
        cl_G  = l_emu.predict(cosmo_6d, Deltaz_4d)
    """

    _FIELD_CONFIG = {
        'L_fid':                  (True,  4),
        'sqrt_L_diag_fid':        (True,  4),
        'grad_L':                 (False, 4),
        'grad_sqrt_L_diag':       (False, 4),
        'hess_L_00':              (False, 4),
        'hess_L_11':              (False, 4),
        'hess_L_10':              (False, 4),
        'hess_sqrt_L_diag_00':    (False, 4),
        'hess_sqrt_L_diag_11':    (False, 4),
        'hess_sqrt_L_diag_10':    (False, 4),
    }

    def __init__(self, stack_path):
        f = np.load(stack_path)
        cosmo_all      = f['cosmo_params']
        self.cosmo_all = cosmo_all
        self.IDs       = f['IDs']
        self.N_bins    = int(f['L_fid'].shape[1])
        self.cosmology_min = cosmo_all.min(axis=0)            # <-- new
        self.cosmology_max = cosmo_all.max(axis=0)            # <-- new
        self.emulators = {}
        for field, (log_t, n_pca) in self._FIELD_CONFIG.items():
            data = f[field]
            mask = self._structural_mask(field, data.shape[1:])
            emu  = MaskedDerivativeEmu((cosmo_all, data), mask=mask,
                                       log_t=log_t, N_PCA=n_pca)
            emu.fit_params()
            self.emulators[field] = emu
        f.close()

    def _structural_mask(self, field, trailing_shape):
        N_bins = self.N_bins
        mask   = np.ones(trailing_shape, dtype=bool)
        if 'sqrt_L_diag' in field:
            return mask
        bin_axes = [a for a, s in enumerate(trailing_shape) if s == N_bins]
        i_ax, j_ax = bin_axes[-2], bin_axes[-1]
        i_idx, j_idx = np.indices((N_bins, N_bins))
        lower_tri = i_idx > j_idx
        shape_for_broadcast = [1] * len(trailing_shape)
        shape_for_broadcast[i_ax] = N_bins
        shape_for_broadcast[j_ax] = N_bins
        lower_tri_full = lower_tri.reshape(shape_for_broadcast)
        mask = mask & ~np.broadcast_to(lower_tri_full, trailing_shape)
        return mask

    def predict(self, cosmo, Deltaz):
        """
        Predict cl_G at (cosmo, Δz). All operations are torch — gradients flow
        back through both inputs.
        """
        if not isinstance(cosmo, torch.Tensor):
            cosmo = torch.tensor(cosmo, dtype=torch.double)
        if not isinstance(Deltaz, torch.Tensor):
            Deltaz = torch.tensor(Deltaz, dtype=torch.double)
        coeffs = {name: emu.predict(cosmo)               # no .detach().numpy()
                  for name, emu in self.emulators.items()}
        return self._cl_G(coeffs, Deltaz)

    def _cl_G(self, c, Deltaz):
        N_bins   = self.N_bins
        diag_idx = torch.arange(N_bins)
        ell_idx  = torch.arange(c['L_fid'].shape[-1])

        L = c['L_fid'].clone()
        L = L + torch.einsum('kijl,k->ijl', c['grad_L'], Deltaz)
        L = L + 0.5 * Deltaz[0]**2 * c['hess_L_00']
        L = L + 0.5 * Deltaz[1]**2 * c['hess_L_11']
        L = L + Deltaz[0] * Deltaz[1] * c['hess_L_10']

        sqrt_diag = (c['sqrt_L_diag_fid']
                    + torch.einsum('kil,k->il', c['grad_sqrt_L_diag'], Deltaz)
                    + 0.5 * Deltaz[0]**2 * c['hess_sqrt_L_diag_00']
                    + 0.5 * Deltaz[1]**2 * c['hess_sqrt_L_diag_11']
                    + Deltaz[0] * Deltaz[1] * c['hess_sqrt_L_diag_10'])

        # Replace the diagonal of L with sqrt_diag**2 via index_put
        # (the slice trick doesn't work in torch, so we build full meshgrid indices)
        i_grid, l_grid = torch.meshgrid(diag_idx, ell_idx, indexing='ij')
        L = L.index_put((i_grid, i_grid, l_grid), sqrt_diag**2)

        return torch.einsum('kil,kjl->ijl', L, L)
    
class DirectEmu:
    """
    Per-scalar quadratic-in-cosmology emulator for small tensors. No PCA.
    Each scalar entry of the trailing shape gets its own ScalarEmulator.

    Skips entries that are zero across all training cosmologies (structural
    zeros from separability — e.g., grad_params[k, i] for k != i).

    Differentiable w.r.t. cosmo through torch.
    """
    def __init__(self, training_data):
        cosmo, values = training_data
        self.trailing_shape = values.shape[1:]
        N_cosmo = values.shape[0]
        flat = values.reshape(N_cosmo, -1).astype(np.float64)
        self.N_flat = flat.shape[1]

        # Identify which entries are structurally zero across all cosmologies
        self.active_mask = np.any(flat != 0, axis=0)         # (N_flat,) bool
        self.N_active    = int(self.active_mask.sum())

        # One ScalarEmulator per active entry
        self.emu = [None] * self.N_flat
        for i in range(self.N_flat):
            if self.active_mask[i]:
                self.emu[i] = ScalarEmulator([cosmo, flat[:, i]])

        # Cached torch tensors for predict
        self.active_idx_t = torch.tensor(np.where(self.active_mask)[0], dtype=torch.long)

    def fit_params(self):
        for emu in self.emu:
            if emu is not None:
                emu.fit_params()
        self.trained = True

    def predict(self, cosmo):
        # Collect predictions for active entries
        active_preds = [emu.predict(cosmo).reshape(()) for emu in self.emu if emu is not None]
        active_preds = torch.stack(active_preds)               # (N_active,)

        # Reinsert into full flat tensor (zeros for inactive entries)
        full_flat = torch.zeros(self.N_flat, dtype=torch.double)
        full_flat = full_flat.index_put((self.active_idx_t,), active_preds)
        return full_flat.reshape(self.trailing_shape)


class params_emu:
    """
    Cosmology emulator for the G_N params Taylor coefficients.
    Per-scalar quadratic-in-cosmology fits; no PCA. Differentiable w.r.t.
    cosmo and Δz.

    Usage:
        p_emu  = params_emu('/path/to/stacked.npz')
        params = p_emu.predict(cosmo_6d, Deltaz_4d)
    """

    _FIELDS = [
        'params_fid', 'grad_params', 'hess_params_00', 'hess_params_11',
        # 'hess_params_10' — structurally zero by separability, not emulated
    ]

    def __init__(self, stack_path):
        f = np.load(stack_path)
        cosmo_all       = f['cosmo_params']
        self.cosmo_all  = cosmo_all
        self.IDs        = f['IDs']
        self.N_bins     = int(f['params_fid'].shape[1])
        self.N_fit      = int(f['params_fid'].shape[2])

        self.emulators = {}
        for field in self._FIELDS:
            emu = DirectEmu((cosmo_all, f[field]))
            emu.fit_params()
            self.emulators[field] = emu
            print(f"  {field}: trained ({emu.N_active}/{emu.N_flat} active entries)")
        f.close()

    def predict(self, cosmo, Deltaz):
        """
        Predict the G_N params at (cosmo, Δz). Returns shape (N_bins, N_fit).
        """
        if not isinstance(cosmo, torch.Tensor):
            cosmo = torch.tensor(cosmo, dtype=torch.double)
        if not isinstance(Deltaz, torch.Tensor):
            Deltaz = torch.tensor(Deltaz, dtype=torch.double)

        c = {name: emu.predict(cosmo) for name, emu in self.emulators.items()}

        # Same Taylor expansion as L_emu but for params:
        #   params(Δz) = params_fid + grad_params · Δz
        #                + ½ Δz_0² hess_00 + ½ Δz_1² hess_11
        # (hess_params_10 = 0)
        out = c['params_fid'].clone()
        out = out + torch.einsum('kjl,k->jl', c['grad_params'], Deltaz)
        out = out + 0.5 * Deltaz[0]**2 * c['hess_params_00']
        out = out + 0.5 * Deltaz[1]**2 * c['hess_params_11']
        return out