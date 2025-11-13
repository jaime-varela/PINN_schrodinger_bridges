import torch
from sklearn import datasets


class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass


class StandardNormalSampler(Sampler):
    def __init__(self, dim=1, device='cuda'):
        super(StandardNormalSampler, self).__init__(device=device)
        self.dim = dim
        
    def sample(self, batch_size=10):
        return torch.randn(batch_size, self.dim, device=self.device)
    
    
class SwissRollSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert dim == 2
        self.dim = 2
        
    def sample(self, batch_size=10):
        batch = datasets.make_swiss_roll(
            n_samples=batch_size,
            noise=0.8
        )[0].astype('float32')[:, [0, 2]] / 7.5
        return torch.tensor(batch, device=self.device)
