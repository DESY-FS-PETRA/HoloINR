import torch
import numpy as np


class FresnelPropagator:
    def __init__(self, fresnel_numbers: float, data_shape: int, downsampling, device):
        self.fresnel_number = fresnel_numbers
        self.device = device
        self.downsampling = downsampling

        data_shape = max(int(1/fresnel_numbers) + 100, data_shape//downsampling + 150)        # data_shape = tuple(np.array(data_shape) + 2*(np.min(data_shape)  - 1))
        self.data_shape = data_shape
        sample_grid = torch.meshgrid(
            torch.fft.fftfreq(self.data_shape, device=device),
            torch.fft.fftfreq(self.data_shape, device=device),
            indexing="ij",
        )
        xi, eta = sample_grid

        self.kernel_func = torch.exp(
            (-1j * np.pi) / self.fresnel_number * (xi * xi + eta * eta)
        )

    def update_fresnel_number(self, fresnel_number):
        self.fresnel_number = fresnel_number
        sample_grid = torch.meshgrid(
            torch.fft.fftfreq(self.data_shape, device=self.device),
            torch.fft.fftfreq(self.data_shape, device=self.device),
            indexing="ij",
        )
        xi, eta = sample_grid

        self.kernel_func = torch.exp(
            (-1j * np.pi) / self.fresnel_number * (xi * xi + eta * eta)
        )

    def propagate(self, x):
        propagated = torch.fft.ifft2(
            torch.fft.fft2(x.reshape(-1, self.data_shape, self.data_shape))
            * self.kernel_func.to(self.device)
        )
        return propagated

    def get_measurements(self, x):
        return torch.abs(self.propagate(x))
