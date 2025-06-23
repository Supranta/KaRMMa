import gpytorch
import torch
import numpy as np
from sklearn.decomposition import PCA

def train_gp(model, likelihood, train_data, training_iter=200, verbose=False):
    train_x, train_y = train_data
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    
    return model.eval(), likelihood.eval()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)        

class ScalarEmulator:
    def __init__(self, data):
        theta, scalar = data
        self.train_x = torch.Tensor(theta).to(torch.double)
        self.SCALAR_MEAN = torch.Tensor([np.mean(scalar, axis=0)]).to(torch.double)
        self.SCALAR_STD  = torch.Tensor([np.std(scalar, axis=0)]).to(torch.double)
        self.scalar_norm = (torch.Tensor(scalar).to(torch.double) - self.SCALAR_MEAN) / self.SCALAR_STD
        
    def train_emu(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.train_x, self.scalar_norm, likelihood)
        model, likelihood = train_gp(model, 
                                     likelihood, [self.train_x, self.scalar_norm])
        self.model      = model
        self.likelihood = likelihood
        self.trained = True
        
    def predict_emu(self, theta_pred):
        scalar_norm_pred = self.likelihood(self.model(theta_pred)).mean
        scalar_pred = self.SCALAR_MEAN + self.SCALAR_STD * scalar_norm_pred
        return scalar_pred
        
class TomographicEmulator:  
    def __init__(self, data):        
        theta, y_tomo = data
        self.N_Z_BINS = y_tomo.shape[1]
        self.y_emu = [ScalarEmulator([theta, y_tomo[:,i]]) for i in range(self.N_Z_BINS)]

    def train_emu(self):
        for i in range(self.N_Z_BINS):
            self.y_emu[i].train_emu()
        self.trained = True
        
    def predict_emu(self, theta_pred):
        y_pred = torch.zeros(self.N_Z_BINS)
        for i in range(self.N_Z_BINS):
            y_pred[i] = self.y_emu[i].predict_emu(theta_pred)
        return y_pred
    
class ClEmu:
    def __init__(self, data, N_PCA):
        theta, cl = data
        self.N_PCA = N_PCA
        log_cl = np.log(cl.reshape((cl.shape[0], -1)) + 1e-25)
        pca = PCA(N_PCA)
        pca.fit(log_cl)
        pca_coeff = pca.transform(log_cl)
        self.pca_coeff_emulators = [ScalarEmulator([theta, pca_coeff[:,i]]) for i in range(N_PCA)]
        
        self.pca_mean = torch.Tensor(pca.mean_).to(torch.double)
        self.pca_components = torch.Tensor(pca.components_[:,np.newaxis])
    
    def train_emu(self):
        for i in range(self.N_PCA):
            print("Training GP # %d"%(i+1))
            self.pca_coeff_emulators[i].train_emu()
        self.trained = True
        
    def predict_emu(self, theta_pred):
        log_cl_pred = self.pca_mean
        for i in range(self.N_PCA):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pca_pred_i = self.pca_coeff_emulators[i].predict_emu(theta_pred)
                log_cl_pred = log_cl_pred + pca_pred_i * self.pca_components[i]
        log_cl_pred = log_cl_pred[0].reshape((4,4,-1))
        return torch.exp(log_cl_pred)