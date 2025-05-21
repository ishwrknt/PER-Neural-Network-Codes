# import numpy as np
# import torch
# import torch.nn as nn
# from PIL import Image
# import matplotlib.pyplot as plt
# import os

# # Parameters (dimensionless units)
# L = 1.0  # Width of the box (dimensionless, equivalent to setting L = 1)
# hbar = 1.0  # Reduced Planck constant (dimensionless)
# m = 1.0  # Particle mass (dimensionless)
# n = 1  # Quantum number for ground state
# E = (n * np.pi) ** 2 / (2 * m * L ** 2)  # Energy eigenvalue (dimensionless)

# # ===================================================================================================== #
# # Analytical solution for the time-independent wave function
# def analytical_wavefunction(x, n=1, L=1.0):
#     return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

# # Generate spatial points for the full domain (for plotting and prediction)
# x = np.linspace(0, L, 100)
# x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
# psi_analytical = torch.tensor(analytical_wavefunction(x.numpy(), n, L), dtype=torch.float32).view(-1, 1)

# # Training data (10 points in [0, L/4])
# L_initial = L / 4  # Restrict to x in [0, 0.25]
# x_data = torch.linspace(0, L_initial, 10, dtype=torch.float32).view(-1, 1)  # 10 points in [0, 0.25]
# psi_data = torch.tensor(analytical_wavefunction(x_data.numpy(), n, L), dtype=torch.float32).view(-1, 1)

# print(x.shape, x_data.shape, psi_data.shape)

# # Plot analytical solution with training points
# plt.figure(figsize=(10, 5))
# plt.plot(x, analytical_wavefunction(x.numpy(), n, L), label='Analytical wave function (n=1)', color='red')
# plt.scatter(x_data, psi_data, color='orange', label='Training points', s=60, alpha=0.4)
# plt.xlabel('x', fontsize=25)
# plt.ylabel(r'$\psi(x)$', fontsize=25)
# plt.tick_params(labelsize=16)
# plt.legend(frameon=False, fontsize=16)
# plt.title('Ground State (n=1) Wave Function', fontsize=20)
# plt.tight_layout()
# plt.savefig('analytical_wavefunction_n1.svg', bbox_inches='tight', pad_inches=0.1)
# plt.show()

# # ===================================================================================================== #
# # Neural Network Definition
# class FCN(nn.Module):
#     "Defines a fully connected network"
    
#     def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
#         super().__init__()
#         activation = nn.Tanh
#         self.fcs = nn.Sequential(*[
#                         nn.Linear(N_INPUT, N_HIDDEN),
#                         activation()])
#         self.fch = nn.Sequential(*[
#                         nn.Sequential(*[
#                             nn.Linear(N_HIDDEN, N_HIDDEN),
#                             activation()]) for _ in range(N_LAYERS-1)])
#         self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
#     def forward(self, x):
#         x = self.fcs(x)
#         x = self.fch(x)
#         x = self.fce(x)
#         return x

# # ===================================================================================================== #
# def save_gif_PIL(outfile, files, fps=5, loop=0):
#     "Helper function for saving GIFs"
#     imgs = [Image.open(file) for file in files]
#     imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

# def plot_result(x, psi_analytical, x_data, psi_data, psi_pred, step=0):
#     "Plot training results for time-independent wave function"
#     plt.figure(figsize=(8, 4))
#     plt.plot(x, psi_analytical, color="tab:green", linewidth=2, alpha=0.8, label="Analytical (n=1)")
#     plt.plot(x, psi_pred, color="tab:blue", linewidth=4, alpha=0.8, label="PINN prediction")
#     plt.scatter(x_data, psi_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
#     plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
#     plt.xlim(-0.05, L + 0.05)
#     plt.ylim(-1.5, 1.5)
#     plt.text(1.065, 0.7, f"Training step: {step}", fontsize="xx-large", color="k")
#     plt.xlabel('x', fontsize=16)
#     plt.ylabel(r'$\psi(x)$', fontsize=16)
#     plt.title('Time-Independent Wave Function (n=1)', fontsize=16)
#     plt.tight_layout()

# # ===================================================================================================== #
# # Physics-Informed Neural Network (PINN)
# x_physics = torch.linspace(0, L, 30).view(-1, 1).requires_grad_(True)
# model = FCN(1, 1, 32, 3)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# if not os.path.exists('plots'):
#     os.makedirs('plots')

# files = []
# data_loss_weight = 1e2  # Weight to prioritize data loss
# for i in range(20000):
#     optimizer.zero_grad()
    
#     # Compute the data loss
#     psi_h = model(x_data)
#     loss1 = torch.mean((psi_h - psi_data) ** 2)
    
#     # Compute the physics loss (Schrödinger equation: -ħ²/(2m) d²ψ/dx² = Eψ)
#     psi_hp = model(x_physics)
#     dpsi_dx = torch.autograd.grad(psi_hp, x_physics, torch.ones_like(psi_hp), create_graph=True)[0]
#     d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_physics, torch.ones_like(dpsi_dx), create_graph=True)[0]
#     physics = -(hbar ** 2 / (2 * m)) * d2psi_dx2 - E * psi_hp  # Time-independent Schrödinger equation
#     loss2 = (1e-4) * torch.mean(physics ** 2)
    
#     # Boundary conditions: ψ(0) = ψ(L) = 0
#     psi_bc0 = model(torch.tensor([[0.0]], requires_grad=True))
#     psi_bcL = model(torch.tensor([[L]], requires_grad=True))
#     loss3 = torch.mean(psi_bc0 ** 2) + torch.mean(psi_bcL ** 2)
    
#     # Total loss with weighted data loss
#     loss = data_loss_weight * loss1 + loss2 + loss3
#     loss.backward()
#     optimizer.step()
    
#     # Print loss values to diagnose
#     if (i + 1) % 150 == 0:
#         print(f"Step {i+1}: loss1 (data) = {loss1.item():.6f}, loss2 (physics) = {loss2.item():.6f}, "
#               f"loss3 (boundary) = {loss3.item():.6f}, total loss = {loss.item():.6f}")
        
#         psi_pred = model(x).detach().numpy()
#         plot_result(x, psi_analytical.numpy(), x_data, psi_data, psi_pred, step=i + 1)
        
#         file = f"plots/pinn_wavefunction_n1_%.8i.png" % (i + 1)
#         plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
#         files.append(file)
        
#         if (i + 1) % 6000 == 0:
#             plt.show()
#         else:
#             plt.close("all")

# save_gif_PIL("pinn_wavefunction_n1.gif", files, fps=20, loop=0)






import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os

# Parameters (dimensionless units)
L = 1.0  # Width of the box (dimensionless, equivalent to setting L = 1)
hbar = 1.0  # Reduced Planck constant (dimensionless)
m = 1.0  # Particle mass (dimensionless)
n = 1  # Quantum number for ground state
E = (n * np.pi) ** 2 / (2 * m * L ** 2)  # Energy eigenvalue (dimensionless)

# ===================================================================================================== #
# Analytical solution for the time-independent wave function
def analytical_wavefunction(x, n=1, L=1.0):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

# Generate spatial points for the full domain (for plotting and prediction)
x = np.linspace(0, L, 100)
x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
psi_analytical = torch.tensor(analytical_wavefunction(x.numpy(), n, L), dtype=torch.float32).view(-1, 1)

# Training data (10 points in [0, L/4])
L_initial = L / 4  # Restrict to x in [0, 0.25]
x_data = torch.linspace(0, L_initial, 10, dtype=torch.float32).view(-1, 1)  # 10 points in [0, 0.25]
psi_data = torch.tensor(analytical_wavefunction(x_data.numpy(), n, L), dtype=torch.float32).view(-1, 1)

print(x.shape, x_data.shape, psi_data.shape)

# Plot analytical solution with training points
plt.figure(figsize=(10, 5))
plt.plot(x, analytical_wavefunction(x.numpy(), n, L), label='Analytical wave function (n=1)', color='red')
plt.scatter(x_data, psi_data, color='orange', label='Training points', s=60, alpha=0.4)
plt.xlabel('x', fontsize=25)
plt.ylabel(r'$\psi(x)$', fontsize=25)
plt.tick_params(labelsize=16)
plt.legend(frameon=False, fontsize=16)
plt.title('Ground State (n=1) Wave Function', fontsize=20)
plt.tight_layout()
plt.savefig('analytical_wavefunction_n1.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()

# ===================================================================================================== #
# Neural Network Definition
class FCN(nn.Module):
    "Defines a fully connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# ===================================================================================================== #
def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def plot_result(x, psi_analytical, x_data, psi_data, psi_pred, step=0):
    "Plot training results for time-independent wave function"
    plt.figure(figsize=(8, 4))
    plt.plot(x, psi_analytical, color="tab:green", linewidth=2, alpha=0.8, label="Analytical (n=1)")
    plt.plot(x, psi_pred, color="tab:blue", linewidth=4, alpha=0.8, label="PINN prediction")
    plt.scatter(x_data, psi_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.xlim(-0.05, L + 0.05)
    plt.ylim(-1.5, 1.5)
    plt.text(1.065, 0.7, f"Training step: {step}", fontsize="xx-large", color="k")
    plt.xlabel('x', fontsize=16)
    plt.ylabel(r'$\psi(x)$', fontsize=16)
    plt.title('Time-Independent Wave Function (n=1)', fontsize=16)
    plt.tight_layout()

# ===================================================================================================== #
# Physics-Informed Neural Network (PINN)
x_physics = torch.linspace(0, L, 30).view(-1, 1).requires_grad_(True)
model = FCN(1, 1, 32, 3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

if not os.path.exists('plots'):
    os.makedirs('plots')

files = []
data_loss_weight = 1e2  # Weight to prioritize data loss
for i in range(20000):
    optimizer.zero_grad()
    
    # Compute the data loss
    psi_h = model(x_data)
    loss1 = torch.mean((psi_h - psi_data) ** 2)
    
    # Compute the physics loss (Schrödinger equation: -ħ²/(2m) d²ψ/dx² = Eψ)
    psi_hp = model(x_physics)
    dpsi_dx = torch.autograd.grad(psi_hp, x_physics, torch.ones_like(psi_hp), create_graph=True)[0]
    d2psi_dx2 = torch.autograd.grad(dpsi_dx, x_physics, torch.ones_like(dpsi_dx), create_graph=True)[0]
    physics = -(hbar ** 2 / (2 * m)) * d2psi_dx2 - E * psi_hp  # Time-independent Schrödinger equation
    loss2 = (1e-4) * torch.mean(physics ** 2)
    
    # Boundary conditions: ψ(0) = ψ(L) = 0
    psi_bc0 = model(torch.tensor([[0.0]], requires_grad=True))
    psi_bcL = model(torch.tensor([[L]], requires_grad=True))
    loss3 = torch.mean(psi_bc0 ** 2) + torch.mean(psi_bcL ** 2)
    
    # Normalization condition: ∫ψ^2 dx = 1
    dx = x_physics[1] - x_physics[0]
    normalization_integral = torch.sum(psi_hp ** 2) * dx  # ∫ ψ^2 dx
    loss4 = (1e-4) * (normalization_integral - 1.0) ** 2  # Penalize deviation from 1
    
    # Total loss with weighted data loss and normalization
    loss = data_loss_weight * loss1 + loss2 + loss3 + loss4
    loss.backward()
    optimizer.step()
    
    # Print loss values to diagnose
    if (i + 1) % 150 == 0:
        print(f"Step {i+1}: loss1 (data) = {loss1.item():.6f}, loss2 (physics) = {loss2.item():.6f}, "
              f"loss3 (boundary) = {loss3.item():.6f}, loss4 (normalization) = {loss4.item():.6f}, "
              f"total loss = {loss.item():.6f}")
        
        psi_pred = model(x).detach().numpy()
        plot_result(x, psi_analytical.numpy(), x_data, psi_data, psi_pred, step=i + 1)
        
        file = f"plots/pinn_wavefunction_n1_%.8i.png" % (i + 1)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        
        if (i + 1) % 6000 == 0:
            plt.show()
        else:
            plt.close("all")

save_gif_PIL("pinn_wavefunction_n1.gif", files, fps=20, loop=0)