import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.init as init

# Ensure CUDA is used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters (dimensionless units)
L = 1.0  # Width of the box
hbar = 1.0  # Reduced Planck constant
m = 1.0  # Particle mass
n = 2  # Quantum number for first excited state
E = (n * np.pi) ** 2 / (2 * m * L ** 2)  # Energy eigenvalue
print(f"Energy eigenvalue: {E:.6f}")

# Analytical solution for the time-independent wave function
def analytical_wavefunction(x, n=2, L=1.0):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

# Generate spatial points
x = np.linspace(0, L, 100)
x = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
psi_analytical = torch.tensor(analytical_wavefunction(x.cpu().numpy(), n, L), dtype=torch.float32).view(-1, 1).to(device)

# Plot analytical solution
plt.figure(figsize=(10, 5))
plt.plot(x.cpu(), analytical_wavefunction(x.cpu().numpy(), n, L), label=f'Analytical wave function (n={n})', color='red')
plt.xlabel('x', fontsize=25)
plt.ylabel(r'$\psi(x)$', fontsize=25)
plt.tick_params(labelsize=16)
plt.legend(frameon=False, fontsize=16)
plt.title(f'First Excited State (n={n}) Wave Function', fontsize=20)
plt.tight_layout()
try:
    plt.savefig('analytical_wavefunction_n2.svg', bbox_inches='tight', pad_inches=0.1)
except Exception as e:
    print(f"Error saving analytical plot: {e}")
plt.show()

# Neural Network Definition
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN), activation()])
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='tanh', a=0.1)
                if m.bias is not None:
                    init.zeros_(m.bias)
        
    def forward(self, x_normalized):
        x_physical = (x_normalized + 1) * L / 2
        x = self.fcs(x_normalized)
        x = self.fch(x)
        x = self.fce(x)
        x_scaled = torch.sin(n * np.pi * x_physical / L) * x + 0.02 * torch.sin(n * np.pi * x_physical / L)  # Adjusted for n=2
        return x_scaled

# Helper function for saving GIFs
def save_gif_PIL(outfile, files, fps=5, loop=0):
    if not files:
        print("No files to create GIF; skipping GIF creation.")
        return
    try:
        imgs = [Image.open(file) for file in files]
        imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
    except Exception as e:
        print(f"Error saving GIF: {e}")

def plot_result(x, psi_analytical, psi_pred, step=0):
    plt.figure(figsize=(8, 4))
    plt.plot(x.cpu().numpy(), psi_analytical.cpu().numpy(), color="tab:green", linewidth=2, alpha=0.8, label=f"Analytical (n={n})")
    plt.plot(x.cpu().numpy(), psi_pred, color="tab:blue", linewidth=4, alpha=0.8, label="PINN prediction")
    plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.xlim(-0.05, L + 0.05)
    plt.ylim(-1.5, 1.5)  # Same range, as amplitude is unchanged
    plt.text(1.065, 0.7, f"Training step: {step}", fontsize="xx-large", color="k")
    plt.xlabel('x', fontsize=16)
    plt.ylabel(r'$\psi(x)$', fontsize=16)
    plt.title(f'Time-Independent Wave Function (n={n})', fontsize=16)
    plt.tight_layout()

# Helper function for computing derivatives
def compute_derivatives(model, x, device):
    x = x.requires_grad_(True)
    psi = model(x)
    dpsi_dx = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    d2psi_dx2 = torch.autograd.grad(dpsi_dx.sum(), x, create_graph=True)[0]
    return psi, dpsi_dx, d2psi_dx2

# Physics-Informed Neural Network (PINN)
x_physics = torch.linspace(0, L, 1000).view(-1, 1).to(device)
x_physics_normalized = 2 * (x_physics / L) - 1
x_normalized = 2 * (x / L) - 1
boundary_points = torch.tensor([[-1.0], [1.0]], dtype=torch.float32, device=device)

model = FCN(1, 1, 32, 3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.7)

if not os.path.exists('plots'):
    try:
        os.makedirs('plots')
    except Exception as e:
        print(f"Error creating plots directory: {e}")

# Plot initial prediction
try:
    psi_pred_initial = model(x_normalized).detach()
    print(f"Initial prediction shape: {psi_pred_initial.shape}")
    plot_result(x, psi_analytical, psi_pred_initial.cpu().numpy(), step=0)
    plt.savefig('plots/pinn_wavefunction_n2_initial.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()
except Exception as e:
    print(f"Error in initial prediction or plotting: {e}")

files = []
for i in range(18000):  # Increased to allow convergence around 20,000 iterations
    try:
        optimizer.zero_grad()
        
        # Physics loss
        psi_hp, dpsi_dx, d2psi_dx2 = compute_derivatives(model, x_physics_normalized, device)
        dx_dx_normalized = L / 2
        dpsi_dx = dpsi_dx / dx_dx_normalized
        d2psi_dx2 = d2psi_dx2 / (dx_dx_normalized ** 2)
        physics = -(hbar ** 2 / (2 * m)) * d2psi_dx2 - E * psi_hp
        loss2 = 0.05 * torch.mean(physics ** 2)
        
        # Normalization loss
        dx = x_physics[1] - x_physics[0]
        normalization_integral = torch.sum(psi_hp ** 2) * dx
        loss4 = (normalization_integral - 1.0) ** 2
        
        # Boundary loss
        # psi_boundary = model(boundary_points)
        # loss_boundary = torch.mean(psi_boundary ** 2)
        
        # Total loss
        # l2_reg = sum(p.norm() ** 2 for p in model.parameters())
        loss = loss2 + 5.0 * loss4 #+ 0.1 * loss_boundary #+ 5e-4 * l2_reg
        
        # Check for numerical stability
        if not torch.isfinite(loss):
            print(f"Loss became non-finite at step {i+1}. Stopping training.")
            break
        
        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.005)
        
        optimizer.step()
        scheduler.step()
        
        # Logging and plotting
        if (i + 1) % 150 == 0 or i < 100:  # Log first 100 steps
            psi_pred = model(x_normalized).detach()
            max_amplitude = torch.max(torch.abs(psi_pred)).item()
            print(f"Step {i+1}: loss2 (physics) = {loss2.item():.6f}, "
                  f"loss4 (normalization) = {loss4.item():.6f}, "
                  #f"loss_boundary = {loss_boundary.item():.6f}, "
                  f"total loss = {loss.item():.6f}, "
                  f"grad_norm = {grad_norm:.6f}, "
                  f"max_amplitude = {max_amplitude:.6f}")
            if max_amplitude < 0.1:
                print(f"Warning: max_amplitude is very low at step {i+1}. Model may be stuck.")
            
            if (i + 1) % 150 == 0:
                psi_pred_np = psi_pred.cpu().numpy()
                plot_result(x, psi_analytical, psi_pred_np, step=i + 1)
                file = f"plots/pinn_wavefunction_n2_%.8i.png" % (i + 1)
                plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
                files.append(file)
                
                if (i + 1) % 6000 == 0:
                    plt.show()
                else:
                    plt.close("all")
                
    except Exception as e:
        print(f"Error at step {i+1}: {e}")
        break

# Save GIF
save_gif_PIL("pinn_wavefunction_n2.gif", files, fps=20)