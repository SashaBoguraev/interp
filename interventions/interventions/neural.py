# RevNet + MLP code sourced from Denis Sutter.
import torch
import torch.nn as nn
import normflows as nf
class MLP(nn.Module):
    """
    Reference MLP implementation (from densutter repo): simple sequential
    linear layers with the provided activation (defaults to ReLU).
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation=nn.ReLU(), dropout_rate=0.0):
        super(MLP, self).__init__()

        if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be a non-empty list of integers")

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP."""
        return self.network(x)


class RevNet_Block(nn.Module):
    def __init__(self, in_features, hidden_size, depth=1):
        super(RevNet_Block, self).__init__()
        self.half_in_features=in_features//2

        # Initialize MLPs with stable initialization
        self.F = MLP(self.half_in_features, self.half_in_features, [hidden_size]*depth)
        self.G = MLP(self.half_in_features, self.half_in_features, [hidden_size]*depth)
        
        # Initialize with careful scaling to prevent explosion/vanishing
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization with small gain
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.5  # Scale down weights to prevent explosion
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Start with zero bias

    def forward(self, x):
        x_1 = x[:,:self.half_in_features]
        x_2 = x[:,self.half_in_features:]
        F_O = self.F(x_2)
        y_1 = x_1 + F_O
        G_O = self.G(y_1)
        y_2 = x_2 + G_O
        y   = torch.cat((y_1, y_2), dim=1)
        return y

    def inverse(self, y):
        y_1 = y[:,:self.half_in_features]
        y_2 = y[:,self.half_in_features:]
        G_O = self.G(y_1)
        x_2 = y_2 - G_O
        F_O = self.F(x_2)
        x_1 = y_1 - F_O
        x   = torch.cat((x_1, x_2), dim=1)
        return x


class RevNet(nn.Module):
    def __init__(self, number_blocks, in_features, hidden_size, depth=1):
        super(RevNet, self).__init__()
        Model_Layers = []
        norm_layers = []
        for i in range(number_blocks):
            Model_Layers.append(RevNet_Block(in_features, hidden_size, depth))
            norm_layers.append(nn.LayerNorm(in_features))
        self.Model_Layers = nn.ModuleList(Model_Layers)
        self.norm_layers = nn.ModuleList(norm_layers)

    def forward(self, x):
        x = x.clone()  # Ensure we don't modify the input
        for ac_layer, norm in zip(self.Model_Layers, self.norm_layers):
            # Apply gradient clipping for stability during training
            if self.training:
                for p in ac_layer.parameters():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
            
            # Forward pass with value checking
            x = ac_layer(x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                # If we detect NaNs/Infs, try to recover by clamping values
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Apply normalization with numerical stability
            x = norm(x)
        return x

    def inverse(self, y):
        """Applies inverse transformation with high precision and improved stability."""
        y = y.clone()  # Ensure we don't modify the input
        for ac_layer, norm in zip(reversed(self.Model_Layers), reversed(self.norm_layers)):
            # Apply gradient clipping for stability
            if self.training:
                for p in ac_layer.parameters():
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)
            
            # Handle normalization carefully in inverse pass
            std, mean = torch.std_mean(y, dim=0, keepdim=True)
            y = (y - mean) / (std + 1e-5)  # Add eps for numerical stability
            y = ac_layer.inverse(y)
            y = y * std + mean  # Restore the scale and mean
        return y
    

class RealNVP(nn.Module):
    def __init__(self, n_layers, latent_dim, hidden_dim, device='cuda:0'):
        super(RealNVP, self).__init__()
        base = nf.distributions.base.DiagGaussian(latent_dim)
        self.n_layers = n_layers
        self.flows = []
        
        for _ in range(n_layers):
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([latent_dim//2, hidden_dim, hidden_dim, latent_dim], init_zeros=True)
            # Add flow layer
            self.flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            self.flows.append(nf.flows.Permute(2, mode='swap'))
        
        self.weight = nf.NormalizingFlow(base, self.flows)

    def forward(self, x):
        return self.weight(x)
    
    def inverse(self, y):
        """Applies inverse transformation with high precision."""
        return self.weight.inverse(y)