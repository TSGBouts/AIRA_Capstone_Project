import torch
from torch import nn
import torch.nn.functional as F

class IKNet(nn.Module):
    """
    IK Neural Network with gated residual blocks and dropout.
    Inputs (6): [x_n, y_n, z_n, r_n, sin(phi), cos(phi)]
    Outputs (6): [sin(q1), cos(q1), sin(q2), cos(q2), sin(q3), cos(q3)]
    This network has multiple deep residual blocks using Gated Linear Units (GLU)
    for improved feature gating, along with LayerNorm and Dropout for stability.
    """
    def __init__(self, in_features: int = 6, hidden: int = 1024, n_blocks: int = 6, dropout_p: float = 0.1):
        super().__init__()
        self.act = nn.SiLU()  # Using SiLU (Swish) activations for smooth gradients
        self.fc_in = nn.Linear(in_features, hidden)
        self.dropout_in = nn.Dropout(dropout_p)

        # Create gated residual blocks
        self.blocks_fc = nn.ModuleList([nn.Linear(hidden, 2 * hidden) for _ in range(n_blocks)])
        self.blocks_ln = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_blocks)])
        self.blocks_do = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(n_blocks)])

        # Output layer to produce 6 values (3 joints * (sin, cos))
        self.fc_out = nn.Linear(hidden, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial transform + activation + input dropout
        out = self.act(self.fc_in(x))
        out = self.dropout_in(out)

        # Apply each gated residual block
        for fc, ln, do in zip(self.blocks_fc, self.blocks_ln, self.blocks_do):
            y = fc(out)
            # GLU: split into value and gate, compute value * sigmoid(gate)
            y = F.glu(y, dim=1)  # after GLU, shape = (batch, hidden)
            y = ln(y)
            y = do(y)
            # Residual addition and activation
            out = self.act(out + y)

        # Final output layer
        y = self.fc_out(out)               # shape (batch, 6)
        y = y.view(-1, 3, 2)              # shape (batch, 3, 2)
        # Normalize each (sin, cos) pair to unit length for stability
        y = y / (torch.norm(y, dim=2, keepdim=True) + 1e-8)
        return y.view(-1, 6)              # shape (batch, 6)

def decode_angles(sin_cos: torch.Tensor) -> torch.Tensor:
    """
    Convert network output (batch,6) into (batch,3) joint angles using atan2.
    Input sin_cos tensor is shaped (batch,6), representing (sin(q1),cos(q1),...,sin(q3),cos(q3)).
    """
    sc = sin_cos.view(-1, 3, 2)
    s = sc[:, :, 0]  # sin values for q1,q2,q3
    c = sc[:, :, 1]  # cos values for q1,q2,q3
    # atan2 returns angles in radians in range [-pi, pi]
    return torch.atan2(s, c)
