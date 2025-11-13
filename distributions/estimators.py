"""
    Tools for estimating the Bridge boundary distribution from samples.
"""
import torch
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor



from sklearn.preprocessing import StandardScaler

class BayesianGMM:
    def __init__(self, X_0, X_1, n_components=32, max_iter=1000, use_pca=False):
        self.scaler0 = StandardScaler().fit(X_0)
        Z0 = self.scaler0.transform(X_0)

        if use_pca:
            self.pca0 = PCA(n_components=128).fit(Z0)
            Z0 = self.pca0.transform(Z0)
        else:
            self.pca0 = None

        self.bgmm_zero = BayesianGaussianMixture(
            n_components=n_components, covariance_type="diag",
            weight_concentration_prior=1.0/n_components, max_iter=max_iter)
        self.bgmm_zero.fit(Z0)

        self.scaler1 = StandardScaler().fit(X_1)
        Z1 = self.scaler1.transform(X_1)

        if use_pca:
            self.pca1 = PCA(n_components=128).fit(Z1)
            Z1 = self.pca1.transform(Z1)
        else:
            self.pca1 = None

        self.bgmm_one = BayesianGaussianMixture(
            n_components=n_components, covariance_type="diag",
            weight_concentration_prior=1.0/n_components, max_iter=max_iter)
        self.bgmm_one.fit(Z1)

    def _prep0(self, x: Tensor) -> np.ndarray:
        X = x.detach().cpu().numpy()
        Z = self.scaler0.transform(X)
        if self.pca0 is not None:
            Z = self.pca0.transform(Z)
        return Z

    def _prep1(self, x: Tensor) -> np.ndarray:
        X = x.detach().cpu().numpy()
        Z = self.scaler1.transform(X)
        if self.pca1 is not None:
            Z = self.pca1.transform(Z)
        return Z

    def rho0(self, x: Tensor) -> Tensor:
        logp = self.bgmm_zero.score_samples(self._prep0(x))
        return torch.from_numpy(np.exp(logp)).to(x).reshape(-1)

    def rho1(self, x: Tensor) -> Tensor:
        logp = self.bgmm_one.score_samples(self._prep1(x))
        return torch.from_numpy(np.exp(logp)).to(x).reshape(-1)
