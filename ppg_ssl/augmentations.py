import numpy as np
import torch

class PPGTransforms:
    def __init__(self, fs=64.0, domain_type='wearable'):
        self.fs = fs
        self.domain_type = domain_type
        # Set parameters based on domain
        if domain_type == 'wearable':
            self.sigma_jitter = 0.02
            self.sigma_scaling = 0.2
            self.wander_prob = 0.5
        else: # clinical
            self.sigma_jitter = 0.005
            self.sigma_scaling = 0.05
            self.wander_prob = 0.0 # Disable wander for clinical

    def jitter(self, x, sigma=None):
        if sigma is None: sigma = self.sigma_jitter
        noise = np.random.normal(0, sigma, x.shape)
        return x + noise

    def scaling(self, x, sigma=None):
        if sigma is None: sigma = self.sigma_scaling
        factor = np.random.normal(1, sigma)
        return x * factor

    def baseline_wander(self, x, intensity=0.1):
        l = len(x)
        freq = np.random.uniform(0.1, 0.4)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.arange(l) / self.fs
        wander = intensity * np.sin(2 * np.pi * freq * t + phase)
        return x + wander.reshape(-1, 1)

    def mask(self, x, mask_prob=0.1):
        x_aug = x.copy()
        mask_len = int(len(x) * mask_prob)
        start = np.random.randint(0, len(x) - mask_len)
        x_aug[start:start+mask_len] = 0
        return x_aug

    def __call__(self, x):
        if torch.is_tensor(x): x = x.numpy()
        x_aug = x.copy()
        if np.random.rand() < 0.8: x_aug = self.jitter(x_aug)
        if np.random.rand() < 0.8: x_aug = self.scaling(x_aug)
        if np.random.rand() < self.wander_prob: x_aug = self.baseline_wander(x_aug)
        if np.random.rand() < 0.3: x_aug = self.mask(x_aug)
        return torch.from_numpy(x_aug.astype(np.float32))

class ContrastiveTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return self.transform(x), self.transform(x)
