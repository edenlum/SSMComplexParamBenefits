import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import wandb
import ray


class VanillaSSM(nn.Module):
    def __init__(self, N, complex_option, BC_std, r_min=0.99):
        super(VanillaSSM, self).__init__()
        self.N = N
        self.complex_option = complex_option
        self.BC_std = BC_std

        if self.complex_option:
            rad_log = self.init_rad_iniform_in_band(N, r_min=r_min)
            angle = torch.rand(N) * 2 * np.pi
            self.rad_log = nn.Parameter(rad_log.unsqueeze(1))
            self.angle = nn.Parameter(angle.unsqueeze(1))

            B_real, B_imag = self.init_complex_normal(N, BC_std)
            self.B_real = nn.Parameter(B_real)
            self.B_imag = nn.Parameter(B_imag)

            C_real, C_imag = self.init_complex_normal(N, BC_std)
            self.C_real = nn.Parameter(C_real)
            self.C_imag = nn.Parameter(C_imag)
        else:
            rad_log = self.init_rad_iniform_in_band(N, r_min=r_min)
            angle = torch.tensor([np.random.choice([0, np.pi], size=N)]).to(torch.float32).squeeze(0)
            self.rad_log = nn.Parameter(rad_log.unsqueeze(1))
            self.angle = nn.Parameter(angle.unsqueeze(1), requires_grad=False)

            self.B = nn.Parameter(torch.randn(N, 1) * BC_std)
            self.C = nn.Parameter(torch.randn(N, 1) * BC_std)

    def init_rad_iniform_in_band(self, size, r_min, r_max=0.9999):
        # rad such that complex are uniform in band
        u = np.random.uniform(size=(size,))
        rad = u * (r_max ** 2 - r_min ** 2) + r_min ** 2
        rad_log = np.log(-0.5 * np.log(rad))
        return torch.tensor(rad_log).to(torch.float32)

    def init_complex_normal(self, size, std):
        angles = torch.rand(size) * 2 * np.pi
        radii = torch.abs(torch.randn(size) * std)
        real_part = radii * torch.cos(angles)
        imag_part = radii * torch.sin(angles)
        return real_part.unsqueeze(1), imag_part.unsqueeze(1)

    def forward(self, x):
        batch_size = x.size(0)
        L = x.size(1)
        if self.complex_option:
            radii = torch.exp(-torch.exp(self.rad_log))
            A_real = radii * torch.cos(self.angle)
            A_imag = radii * torch.sin(self.angle)
            A = A_real + 1j * A_imag
            B = self.B_real + 1j * self.B_imag
            C = self.C_real + 1j * self.C_imag
            h_t = torch.zeros(batch_size, self.B_real.size(0), 1,
                              dtype=torch.complex64)  # Initialize h_t as purely imaginary
        else:
            radii = torch.exp(-torch.exp(self.rad_log))
            A = radii * torch.cos(self.angle)
            B = self.B
            C = self.C
            h_t = torch.zeros(batch_size, self.B.size(0), 1)

        A = A.unsqueeze(0)
        B = B.unsqueeze(0)

        outputs = []
        for t in range(L):
            h_t = A * h_t + B * (x[:, t].view(batch_size, 1, 1))
            output_t = torch.real(torch.matmul(h_t[:, :, 0], C))
            outputs.append(output_t.squeeze(-1))

        outputs = torch.stack(outputs, dim=1)
        if self.complex_option:
            outputs = 2 * outputs

        return outputs


# Define the impulse response function
def impulse_response(model, length):
    impulse = torch.zeros((1, length, 1))
    impulse[0, 0, 0] = 1
    response = model(impulse)
    return response.squeeze()


def get_target_imp_res(task_name, L):
    if task_name == "delay":
        target_imp_res = torch.zeros((L,))
        target_imp_res[(L - 1) // 2] = 1
    elif task_name == "random":
        target_imp_res = (torch.rand(L) * 2 - 1)
    elif task_name == "oscillation":
        target_imp_res = torch.zeros((L,))
        for pow in range(L):
            target_imp_res[pow] = (1j ** pow).real
        return target_imp_res
    else:
        raise Exception("unkown task_name")
    return target_imp_res
