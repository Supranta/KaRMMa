import numpy as np 
from scipy.optimize import minimize
import torch
from sklearn.decomposition import PCA

class ScalarEmulator:
    def __init__(self, training_data):
        self.cosmology_training = training_data[0].numpy()
        self.N_cosmo_params     = self.cosmology_training.shape[1]
        self.N_td               = self.cosmology_training.shape[0]
        self.N                  = self.N_cosmo_params + 1
        self.N_params           = int(0.5*self.N*(self.N+1))
        self.cosmology_mid      = 0.5 * (self.cosmology_training.max(axis=0) + self.cosmology_training.min(axis=0))
        scalar                  = training_data[1].numpy()
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

    
    def cost_function(self,params,cosmo,scalar_obs):
        return np.sum((scalar_obs - self.model(params,cosmo))**2)

    def fit_params(self):
        params      = minimize(self.cost_function,[0 for i in range(self.N_params)],
                                    args=(self.cosmology_training,self.scalar_norm.numpy()[:])).x
        self.zeroth = params[0]
        self.lin    = torch.tensor(params[1:self.N], dtype=torch.double)
        quad_params = params[self.N:]
        self.quad   = torch.tensor(self.symm_matrix(quad_params, self.N_cosmo_params), 
                        dtype=torch.double)   
          
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
    
class LNParamsEmu:
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
        self.emu            = [ScalarEmulator([training_data[0], pca_coeff_norm[:,i]]) for i in range(self.N_PCA)]
   
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
    