import gpytorch
import torch
import numpy as np
from sklearn.decomposition import PCA

def train_gp(model, likelihood, train_data, training_iter=50, verbose=False):
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

class ClEmu:
    def __init__(self, data, N_PCA):
        theta, cl = data
        self.N_PCA = N_PCA
        self.train_x = torch.Tensor(theta).to(torch.double)
        log_cl = np.log(cl.reshape((cl.shape[0], -1)) + 1e-25)
        pca = PCA(N_PCA)
        pca.fit(log_cl)
        pca_coeff = pca.transform(log_cl)
        self.PCA_MEAN = torch.Tensor(np.mean(pca_coeff, axis=0)[np.newaxis]).to(torch.double)
        self.PCA_STD  = torch.Tensor(np.std(pca_coeff, axis=0)[np.newaxis]).to(torch.double)
        self.pca_coeff_norm = (torch.Tensor(pca_coeff).to(torch.double) - self.PCA_MEAN) / self.PCA_STD
        
        self.pca_mean = torch.Tensor(pca.mean_).to(torch.double)
        self.pca_components = torch.Tensor(pca.components_[:,np.newaxis])
    
    def train_emu(self):
        models = []
        likelihoods = []
        for i in range(self.N_PCA):
            print("Training GP # %d"%(i+1))
            train_y = torch.Tensor(self.pca_coeff_norm[:,i])
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(self.train_x, train_y, likelihood)
            model, likelihood = train_gp(model, 
                                         likelihood, 
                                         [self.train_x, train_y])
            models.append(model)
            likelihoods.append(likelihood)
        self.models      = models
        self.likelihoods = likelihoods
        self.trained = True
        
    def predict_emu(self, theta_pred):
        pca_pred = torch.zeros(self.N_PCA, dtype=torch.double)
        log_cl_pred = self.pca_mean
        for i, (model, likelihood) in enumerate(zip(self.models, self.likelihoods)):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pca_coeff_i = likelihood(model(theta_pred)).mean
                pca_pred_i = self.PCA_MEAN[:,i] + self.PCA_STD[:,i] * pca_coeff_i
                log_cl_pred = log_cl_pred + pca_pred_i * self.pca_components[i]
        return pca_coeff_i