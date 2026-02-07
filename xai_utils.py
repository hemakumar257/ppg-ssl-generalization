import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients
        target_layer.register_full_backward_hook(self.save_gradients)
        # Hook for activations
        target_layer.register_forward_hook(self.save_activations)
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def save_activations(self, module, input, output):
        self.activations = output
        
    def __call__(self, x):
        # Forward pass is assumed to be done outside or just before
        # This function computes the heatmap for the *last* forward pass
        
        # 1. Global Average Pooling on Gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])
        
        # 2. Weight activations by pooled gradients
        activations = self.activations[0] # [Channels, Length]
        channels, length = activations.shape
        
        for i in range(channels):
            activations[i, :] *= pooled_gradients[i]
            
        # 3. Average the channels (Heatmap)
        heatmap = torch.mean(activations, dim=0).cpu()
        
        # 4. ReLU (only positive influence)
        heatmap = F.relu(heatmap)
        
        # 5. Normalize
        heatmap /= torch.max(heatmap) + 1e-8
        
        return heatmap.detach().numpy()

def plot_saliency(signal, heatmap, title="Grad-CAM", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    # Upsample heatmap to match signal length
    heatmap_resized = np.interp(
        np.linspace(0, len(heatmap), len(signal)),
        np.arange(len(heatmap)),
        heatmap
    )
    
    # Plot Signal
    ax.plot(signal, 'k', alpha=0.6, label='PPG Signal')
    
    # Overlay Heatmap
    # Create colormap that is transparent for low values
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(1, 0, 0, 0), (1, 0, 0, 0.8)] # Transparent -> Red
    cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=256)
    
    # Fill background based on importance
    extent = [0, len(signal), np.min(signal), np.max(signal)]
    im = ax.imshow(heatmap_resized[np.newaxis, :], cmap=cmap, aspect='auto', extent=extent, alpha=0.7)
    
    ax.set_title(title)
    ax.legend()
    return ax
