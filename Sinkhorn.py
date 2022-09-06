import torch
import numpy as np
import time
import sys

from LogSinkhornGPUBackend import LogSumExpGPU

# Taken from geomloss https://github.com/jeanfeydy/geomloss
def log_dens(α):
    α_log = α.log()
    α_log[α <= 0] = -10000.0
    return α_log

class LogSinkhorn_2D_GPU:
    def __init__(self, mu, nu, eps, alpha_init = None, dx = 1.0,
                 inner_iter = 100, max_iter = 10000, max_error = 1e-4, 
                 max_error_rel = False):
        self.eps = torch.tensor(eps, dtype = torch.float32).item()
        self.dx_eff = torch.tensor(dx/np.sqrt(eps)).item()
        self.dim = len(mu.shape)-1
        assert self.dim == 2, "Softmin not implemented for dim != 2"
        # TODO: implement for other dimension
        # Initialize mu_log, etc
        self.mu = mu
        self.nu = nu
        # print(self.alpha.shape, self.beta.shape)
        self.mu_log = log_dens(mu)
        self.nu_log = log_dens(nu)
        self.shape_alphas = self.mu.shape[1:]
        self.shape_betas = self.nu.shape[1:]
        self.B = self.mu.shape[0]
        # Initialize temp tensors for beta.
        # In case we find that a custom permutation function is better
        B, M, N = self.B, self.shape_alphas, self.shape_betas
        self.beta_partial_1 = torch.cuda.FloatTensor(B*M[0], N[1])
        self.beta_partial_2 = torch.cuda.FloatTensor(B,N[1], M[0])
        self.beta_partial_3 = torch.cuda.FloatTensor(B*N[1], N[0])
        # Initialize temp tensors for alpha
        B, M, N = self.B, self.shape_betas, self.shape_alphas
        self.alpha_partial_1 = torch.cuda.FloatTensor(B*M[0], N[1])
        self.alpha_partial_2 = torch.cuda.FloatTensor(B,N[1], M[0])
        self.alpha_partial_3 = torch.cuda.FloatTensor(B*N[1], N[0])
        # Initialize potentials
        if not alpha_init is None:
            self.alpha = alpha_init
            self.update_beta()
        else:
            self.alpha = torch.zeros_like(mu)
            self.beta = torch.zeros_like(nu)
        # TODO: max_error and max_iter
        self.max_error = max_error
        self.max_error_rel = max_error_rel
        self.inner_iter = inner_iter
        self.max_iter = max_iter


    # Previous version
    def get_new_beta(self):
        B = self.B
        M0, M1 = self.shape_alphas
        N0, N1 = self.shape_betas
        h = self.mu_log + self.alpha / self.eps
        # Do inplace:
        # Softmin on rows
        LogSumExpGPU(h.reshape(B*M0, M1).contiguous(), self.beta_partial_1, self.dx_eff)
        # Reshape and permutedims
        self.beta_partial_2 = self.beta_partial_1.reshape(B, M0, N1).permute((0,2,1)) # TODO: efficient? If not, cuda version
        # Softmin on columns
        LogSumExpGPU(self.beta_partial_2.reshape(B*N1, M0).contiguous(), self.beta_partial_3, self.dx_eff)
        # Reshape and permute dims
        new_beta = self.beta_partial_3.reshape(B, N1, N0).permute((0,2,1))
        # Multiply by epsilon        
        new_beta = new_beta.contiguous()
        return -self.eps * new_beta

    # Test version:
    # def update_beta(self):
    #     B = self.B
    #     M0, M1 = self.shape_alphas
    #     N0, N1 = self.shape_betas
    #     self.alpha = self.mu_log + self.alpha/self.eps
    #     # Do inplace:
    #     # Softmin on rows
    #     # self.alpha.contiguous()
    #     LogSumExpGPU(self.alpha.reshape(B*M0, M1).contiguous(), self.beta_partial_1, self.dx_eff) # TODO: Maybe do contiguous at the end? so that result is contiguous
    #     # Reshape and permutedims
    #     # Softmin on columns
    #     LogSumExpGPU(self.beta_partial_1.reshape(B, M0, N1).permute((0,2,1)).reshape(B*N1, M0).contiguous(), self.beta_partial_3, self.dx_eff)
    #     # Reshape and permute dims
    #     self.beta = -self.eps * self.beta_partial_3.reshape(B, N1, N0).permute((0,2,1)).contiguous()
    #     # Multiply by epsilon
    #     #self.beta *= -self.eps
    
    def get_new_alpha(self):
        B = self.B
        M0, M1 = self.shape_betas
        N0, N1 = self.shape_alphas
        h = self.nu_log + self.beta / self.eps
        # Softmin on rows
        LogSumExpGPU(h.reshape(B*M0, M1).contiguous(), self.alpha_partial_1, self.dx_eff)
        # print(self.alpha_partial_1.reshape(B, M0, N1))
        # Reshape and permutedims
        self.alpha_partial_2 = self.alpha_partial_1.reshape(B, M0, N1).permute((0,2,1)) # TODO: efficient? If not, cuda version
        # Softmin on columns
        # print(self.alpha_partial_2)
        #self.alpha_partial_2.contiguous()
        LogSumExpGPU(self.alpha_partial_2.reshape(B*N1, M0).contiguous(), self.alpha_partial_3, self.dx_eff)
        # Reshape and permute dims
        # print(self.alpha_partial_3.reshape(B, N1, N0))
        new_alpha = self.alpha_partial_3.reshape(B, N1, N0).permute((0,2,1))
        # print(self.alpha)
        # Multiply by epsilon
        new_alpha = new_alpha.contiguous()
        return -self.eps*new_alpha

    def update_beta(self):
        self.beta = self.get_new_beta()

    def update_alpha(self):
        self.alpha = self.get_new_alpha()

    def iterate(self, niter):
        for _ in range(niter-1):
            self.update_alpha()
            self.update_beta()
            # Here one could try different acceleration / stability methods
            # new_alpha = self.get_new_alpha()
            # new_beta = self.get_new_beta()
            # self.alpha = 0.5*(self.alpha + new_alpha)
            # self.beta = 0.5*(self.beta + new_beta)
        beta_prev = torch.clone(self.beta)
        self.update_alpha()
        self.update_beta()
        sinkhorn_error = torch.sum(self.nu*torch.abs(1 - torch.exp((beta_prev - self.beta)/self.eps)))
        return sinkhorn_error

    def iterate_until_max_error(self):
        Niter = 0
        max_error = self.max_error
        if self.max_error_rel:
            max_error *= torch.sum(self.mu)
        current_error = 2*max_error
        while (Niter < self.max_iter) and (current_error >= max_error):
            current_error = self.iterate(self.inner_iter)
            Niter += self.inner_iter
        if current_error < max_error:
            status = 0
        else:
            status = 1
        return status

