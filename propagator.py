from typing import Any
import torch
import torchvision
from fresnel_propagator_torch import FresnelPropagator
from utils import pad, softmin_loss
from holowizard.core.utils.transform import crop_center
import torch.nn.functional as F


class Propagator:
    def __init__(
        self,
        fresnelPropagator: FresnelPropagator,
        logger,
        probes,
        n_angles,
        rayPropagator=None,
        step={"blurred_real": 1.0, "blurred_imag": 1.0},
        downsampling_factor=1,
    ):
        self.gaussianBlurReal = torchvision.transforms.GaussianBlur(
            15, sigma=step["blurred_imag"]
        )
        self.gaussianBlurImag = torchvision.transforms.GaussianBlur(
            15, sigma=step["blurred_real"]
        )
        self.rayPropagator = rayPropagator
        self.fresnelPropagator = fresnelPropagator
        self.loss = dict()
        self.logger = (
            logger  # Placeholder for a writer, e.g., TensorBoard SummaryWriter
        )
        self.probe = probes
        self.n_angles = n_angles
        self._downsampling_factor = downsampling_factor

    @property
    def downsampling_factor(self):
        return self._downsampling_factor

    @downsampling_factor.setter
    def downsampling_factor(self, value):
        self._downsampling_factor = value
        

    def __call__(self, rays, idx, mask=None, log=False, offset=1.0) -> Any:
        """
        Given a set of rays, compute the output image.
        Therefore, we sample the rays and evaluate the network and sum up the result.
        """
        self.loss = dict()
        out = torch.view_as_complex(self.rayPropagator.render(rays, idx, mask=mask))
        out *= offset
          #  out.imag = out.imag - 0.3
        self.l1_loss(out)
        self.TV_loss(out)
        out = pad(out)
        if log:
            self.plot(out)
        out = out 
        out = self.forward(out)
        ###
        if log:
            self.logger.image_from_tensor(out, name="Predicted Hol")

        return out

    def forward(self, out):
        out = torch.exp(1j * out)
        out = self.fresnelPropagator.get_measurements(out).squeeze()
        out = crop_center(out, torch.tensor(self.probe.shape[-2:]) // self.downsampling_factor)
        return out.unsqueeze(0)
    
    def l1_loss(self, y):
        self.loss = dict(**self.loss, L1_real=torch.abs(y.real).mean(), L1_imag=torch.abs(y.imag).mean())

    def smoothness_loss(self, y):
        blurred_real_loss = F.mse_loss(self.gaussianBlurReal(y.real), y.real)
        blurred_imag_loss = F.mse_loss(self.gaussianBlurImag(y.imag), y.imag)
        self.loss = dict(
            **self.loss,
            Smoothness_real=blurred_real_loss,
            Smoothness_imag=blurred_imag_loss,
        )
    def TV_loss(self, y):
        tv_real = torch.mean(torch.abs(y.real[..., :, 1:] - y.real[..., :, :-1])) + torch.mean(torch.abs(y.real[..., 1:, :] - y.real[..., :-1, :]))
        tv_imag = torch.mean(torch.abs(y.imag[..., :, 1:] - y.imag[..., :, :-1])) + torch.mean(torch.abs(y.imag[..., 1:, :] - y.imag[..., :-1, :]))
        self.loss = dict(
            **self.loss,
            TV_real=tv_real,
            TV_imag=tv_imag,
        )

    def plot(self, y):
        self.logger.center_line(y[0].clone().real, "Center Line phase")
        self.logger.center_line(y[0].clone().imag, "Center Line absorption")
        self.logger.image_from_tensor(y[0].clone().imag, name="Absorption")
        self.logger.image_from_tensor(y[0].clone().real, name="Phase")