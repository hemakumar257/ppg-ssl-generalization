import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.layers(x)

class SimCLRWrapper(nn.Module):
    def __init__(self, encoder, input_dim=256):
        super(SimCLRWrapper, self).__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(input_dim=input_dim)
    def forward(self, x1, x2):
        h1 = self.encoder(x1, return_features=True)
        h2 = self.encoder(x2, return_features=True)
        return self.projector(h1), self.projector(h2)

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)
    targets = torch.arange(2 * batch_size, device=z.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size
    return F.cross_entropy(sim_matrix, targets)
