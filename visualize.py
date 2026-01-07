import h5py
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io
import astra

from utils import pad
from cone_beam import ConeBeam
import random
import string

class Logger:
    def __init__(self, writer, downsampling_factor=1):
        self.writer = writer
        self.log = dict()  # Placeholder for a writer, e.g., TensorBoard SummaryWriter
        self.cuda = "cuda:0"
        self.log_epoch = dict()
        self.downsampling_factor = downsampling_factor

    def center_line(self, tensor, name):
        """
        Extract the center line of a 2D tensor and convert it to a NumPy array.
        """

        img = tensor.clone().cpu().detach().numpy()
        center_line = img[img.shape[0] // 2, :]
        fig, ax = plt.subplots()
        g = ax.plot(center_line)
        wandbfig = wandb.Image(fig)
        plt.close(fig)
        self.log[name] = wandbfig

    def line_ploter(self, line, name):
        fig, ax = plt.subplots()
        for i in range(line.shape[0]):
            ax.plot(line[i])
        wandbfig = wandb.Image(fig)
        plt.close(fig)
        self.log[name] = wandbfig

    def image_from_tensor(self, tensor, vmax=None, vmin=None, name="image"):
        """
        Convert a PyTorch tensor to a NumPy array and normalize it to the range [0, 255].
        """
        img = tensor.clone().cpu().detach().numpy().squeeze()
        fig, ax = plt.subplots()
        # Make sure vmax and vmin are defined, otherwise use defaults
        if vmax is None:
            vmax = np.max(img)
        if vmin is None:
            vmin = np.min(img)

        g = ax.imshow(img, cmap="gray", vmax=vmax, vmin=vmin)
        plt.colorbar(g, ax=ax)
        ax.axis("off")
        fig.tight_layout()
        wandbfig = wandb.Image(fig)
        plt.close(fig)
        self.log[name] = wandbfig

    def slice_from_object(self, net):
        with torch.no_grad():
            bound = net.bound // self.downsampling_factor
            XYZ = torch.meshgrid(
                torch.arange(bound, dtype=torch.float32),
                torch.arange(bound, dtype=torch.float32),
                indexing="ij",
            )
            XYZ = torch.stack(XYZ, dim=-1).to(self.cuda)
            XYZ = XYZ.reshape(-1, 2)
            XYZ = XYZ - bound/2
            XYZ = torch.hstack(
                [XYZ, torch.zeros((XYZ.shape[0], 1), dtype=torch.float32).to(self.cuda)]
            )
            net.eval()
            out = net(XYZ)
            out = out.reshape(bound, bound, 2)
            io.imsave("output.tiff", out.cpu().numpy().T)
            self.image_from_tensor(out[..., 0], name="Center Slice phase")
            self.image_from_tensor(out[..., 1], name="Center Slice absorption")
            angular_data_fidelity_values = [
                v for k, v in sorted(self.log_epoch["angular_data_fidelity"].items())
            ]
            angular_data_fidelity_tensor = torch.tensor(angular_data_fidelity_values).unsqueeze(0)
            angular_loss_figure = self.line_ploter(
                angular_data_fidelity_tensor, "Angular Data Fidelity"
            )
            angular_offset_o_values = [
                v for k, v in sorted(self.log_epoch["angular_offset"].items())
            ]
            angular_offset_o_figure = self.line_ploter(
                torch.tensor(angular_offset_o_values).unsqueeze(0),
                "Angular Offset Origin",
            )
            angular_offset_d_values = [
                v for k, v in sorted(self.log_epoch["angular_offset_direction"].items())
            ]
            angular_offset_d_figure = self.line_ploter(
                torch.tensor(angular_offset_d_values).unsqueeze(0),
                "Angular Offset Direction",
            )
            net.train()

        self.log_epoch = {
            "Epoch loss": self.log_epoch["loss"] / 1500,
            "Angular Loss": angular_loss_figure,
            "Angular Offset Origin": angular_offset_o_figure,
            "Angular Offset Direction": angular_offset_d_figure,
            **self.log,
        }
        self.flush_epoch()

    def save_h5(self, net, propagator, downsampling, no_angles):
        length=10
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))

        print("Saving final output to f{}".format(random_string))
        torch.save(net.state_dict(), "f{}.pth".format(random_string))
        with torch.no_grad():
            shape = net.bound // self.downsampling_factor
            shape += 50
            XYZ = torch.meshgrid(torch.arange(shape, dtype=torch.float32), 
                                torch.arange(shape, dtype=torch.float32), 
                                torch.arange(shape, dtype=torch.float32), indexing='ij')
            XYZ = torch.stack(XYZ, dim=-1)
            XYZ = XYZ - shape / 2
            net.eval()
            with torch.no_grad():
                ret = []
                for j in range(shape):
                    slice_data = XYZ[j].to(self.cuda)
                    slice_data = slice_data.reshape(-1, 3)
                    out = net(slice_data).cpu().numpy()
                    out = out.reshape(shape, shape, 2)
                    ret.append(out)
                out = np.stack(ret, axis=0)
                out = out.reshape(shape, shape, shape, 2)
                # get the projections for different angles
                projections = np.zeros((no_angles, shape, shape, 2))
                # Prepare the volume for Radon transform (real and imaginary parts separately)
                volume_real = out[..., 0]
                volume_imag = out[..., 1]

                # ASTRA expects volumes in (z, y, x) order
                volume_real = np.transpose(volume_real, (2, 0, 1))
                volume_imag = np.transpose(volume_imag, (2, 0, 1))

                # Set up ASTRA projection geometry
                angles = np.linspace(0, np.pi, no_angles, endpoint=False)
                proj_geom = proj_geom = astra.create_proj_geom('parallel3d',
                                   1.0,  # detector_spacing_x
                                   1.0,  # detector_spacing_y
                                   shape,  # det_row_count
                                   shape,  # det_col_count
                                   angles)

                # Create ASTRA volume geometries
                vol_geom = astra.create_vol_geom(shape, shape, shape)

                # Compute projections for real part
                proj_id_real, projections_real = astra.create_sino3d_gpu(volume_real, proj_geom, vol_geom)
                astra.data2d.delete(proj_id_real)

                # Compute projections for imaginary part
                proj_id_imag, projections_imag = astra.create_sino3d_gpu(volume_imag, proj_geom, vol_geom)
                astra.data2d.delete(proj_id_imag)

                # projections shape: (no_angles, shape, shape)
                projections[..., 0] = np.transpose(projections_real, (1, 0, 2))
                projections[..., 1] = np.transpose(projections_imag, (1, 0, 2))

                with h5py.File("output_final.h5", "w") as f:
                    f.create_dataset("slices_real", data=np.rollaxis(out[..., 0], 2))
                    f.create_dataset("slices_imag", data=np.rollaxis(out[..., 1], 2))
                    f.create_dataset("projections_real", data=projections[..., 0])
                    f.create_dataset("projections_imag", data=projections[..., 1])

    def flush(self):
        wandb.log(self.log)
        self.log = dict()

    def flush_epoch(self):
        wandb.log(self.log_epoch)
        self.log_epoch = dict()
        self.log = dict()