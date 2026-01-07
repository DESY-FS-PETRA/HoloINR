import torch
import os
import numpy as np
from holowizard.core.utils.remove_outliers import remove_outliers
from holowizard.core.preprocessing.correct_flatfield import correct_flatfield
from holowizard.core.preprocessing.calculate_flatfield_components import calculate_flatfield_components
from torch.utils.data import Dataset
import h5py
from utils import crop
import logging

#  Geometry definition
#
#                  Detector plane, behind
#              |-----------------------------|
#              |                             |
#              |                             |
#              |                             |
#  Centered    |                             |
#    at O      A V    +--------+             |
#              |     /        /|             |
#     A Z      |    /        / |*D           |
#     |        |   +--------+  |             |
#     |        |   |        |  |             |
#     |        |   |     *O |  +             |
#     *--->y   |   |        | /              |
#    /         |   |        |/               |
#   V X        |   +--------+        U       |
#              .--------------------->-------|
#
#            *S
#
#
# An image of the geometry can be found at DOI: 10.1088/2057-1976/2/5/055010
# or simply at: https://i.imgur.com/mRweux3.png
# Geometry class:
#           -nVoxel:        3x1 array of number of voxels in the image
#           -sVoxel:        3x1 array with the total size in mm of the image
#           -dVoxel:        3x1 array with the size of each of the voxels in mm
#           -nDetector:     2x1 array of number of voxels in the detector plane
#           -sDetector:     2x1 array with the total size in mm of the detector
#           -dDetector:     2x1 array with the size of each of the pixels in the detector in mm
#           -DSD:           1x1 or 1xN array. Distance Source Detector, in mm
#           -DSO:           1x1 or 1xN array. Distance Source Origin.
#           -offOrigin:     3x1 or 3xN array with the offset in mm of the centre of the image from the origin.
#           -offDetector:   2x1 or 2xN array with the offset in mm of the centre of the detector from the x axis
#           -rotDetector:   3x1 or 3xN array with the rotation in roll-pitch-yaw of the detector
class ParallelGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter.
    """

    def __init__(self, data):
        ## The following geometry assumes that the image is centered at the origin and that we have parallel beam geometry.
        self.DSO = data["diameter"] / 2
        self.DSD = self.DSO / 2

        # Detector parameters
        self.nDetector = np.array(
            [data["diameter"]] * 2
        )  # number of pixels              (px)
        self.dDetector = [1, 1]  # size of each pixel
        self.sDetector = self.nDetector  # total size of the detector
        # Image parameters
        self.nVoxel = np.array(
            object=data["nVoxels"]
        )  # number of voxels              (vx)
        self.dVoxel = [1, 1, 1]  # size of each voxel. Just assume unit...
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image

        # Offsets ## We assume that the image is centered at the origin and that the plane is at the origin.
        self.offOrigin = np.zeros(3)
        self.offDetector = np.zeros(2)  # Auxiliary

        self.accuracy = data[
            "accuracy"
        ]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = "parallel"  # parallel, cone                ...
        self.filter = data["filter"]

# TODO dumm!

class Preprocessor:
    def __init__(self, data_path, crop_idcs, rank):
        self.data_path = data_path
        self.crop_idcs = crop_idcs
        self.rank = rank
        self.data = self.read_h5_file("data", crop_flag=True)
        self.flatfield = self.read_h5_file("flat", crop_flag=True)
        self.angles = self.read_h5_file("angles", crop_flag=False)
        try:
            self.shifts = self.read_h5_file("shifts", crop_flag=False)
        except KeyError:
            self.shifts = np.zeros((self.data.shape[0], 2))
        self.mask = torch.zeros_like(self.data, dtype=torch.bool)

    def read_h5_file(self, key, crop_flag=False, dtype=np.float32):
        with h5py.File(self.data_path, "r") as f:
            ret = np.squeeze(np.array(f[key][:], dtype=dtype))
            if crop_flag:
                ret = crop(ret, self.crop_idcs)  # - data_dark
            if dtype is bool:
                ret = torch.from_numpy(ret).to(dtype=torch.bool).cpu()
            else:
             #   ret[ret < np.finfo(ret.dtype).eps] = np.finfo(ret.dtype).eps
               # ret = [
               #     remove_outliers(torch.from_numpy(x).to(), threshold=1) for x in ret
               # ]
                ret = torch.from_numpy(ret).cpu()
        return ret


class FlatField(Preprocessor):
    def __init__(self, data_path, n_components, crop_idcs, rank):
        self.data_path = data_path
        self.rank = rank
        self.crop_idcs = crop_idcs
        logging.info("Loading data, flatfields, and mask")
        self.data = self.read_h5_file("data", crop_flag=True)
        self.flatfield = self.read_h5_file("flat", crop_flag=True)
        self.mask = torch.zeros_like(self.data, dtype=torch.bool)
        logging.info("Correcting flatfields")
        components = calculate_flatfield_components(self.flatfield.numpy(), n_components)
        ## TODO probably we have to do this for each angle?
        self.data = torch.stack([correct_flatfield(x, components) for x in self.data.to(self.rank)])
        flatfield_offset_corr = self.data.cpu().mean(axis=(1,2)).numpy()
        self.data = self.data.cpu() / flatfield_offset_corr[:, None, None]
        self.angles = torch.tensor(np.linspace(0, np.pi, self.data.shape[0], endpoint=False))
        try:
            self.shifts = self.read_h5_file("shifts", crop_flag=False)
        except KeyError:
            self.shifts = torch.zeros((self.data.shape[0], 2))
        logging.info("Preprocessing done")


class TIGREDataset(Dataset):
    """
    TIGRE dataset.
    """

    def __init__(
        self,
        data_path,
        preprocessed_path,
        use_every_nth_angle=1,
        crop_idcs=[200, 800, 200, 800],
        rank=0,
        padding_factor=2,
    ):
        super().__init__()
        self.use_every_nth_angle = use_every_nth_angle
        if not os.path.exists(preprocessed_path):
            preprocess = FlatField(data_path, 28, crop_idcs, rank)
            self.projs = preprocess.data
            self.masks = preprocess.mask
            self.probe = preprocess.flatfield
            self.angles = preprocess.angles
            self.shifts = preprocess.shifts
            with h5py.File(preprocessed_path, "w") as f:
                f.create_dataset("data", data=self.projs)
                f.create_dataset("mask", data=self.masks, dtype=bool)
                f.create_dataset("probe", data=self.probe.cpu().numpy())
                f.create_dataset("angles", data=self.angles)
                f.create_dataset("shifts", data=self.shifts)

        else:
            self.projs = torch.from_numpy(h5py.File(preprocessed_path, "r")["data"][:])
            self.masks = torch.from_numpy(h5py.File(preprocessed_path, "r")["mask"][:])
            self.probe = torch.from_numpy(h5py.File(preprocessed_path, "r")["probe"][:])
            self.angles = torch.from_numpy(h5py.File(preprocessed_path, "r")["angles"][:])
            try:
                self.shifts = torch.from_numpy(h5py.File(preprocessed_path, "r")["shifts"][:])
            except KeyError:
                self.shifts = torch.zeros((self.projs.shape[0], 2))


        self.projs = self.projs[:: use_every_nth_angle]
        self.shifts =  np.array(self.shifts[:: use_every_nth_angle])
        self.masks = self.masks[:: use_every_nth_angle]
        self.angles = self.angles[:: use_every_nth_angle] - torch.min(self.angles) 
        self.n_samples = self.projs.shape[0]
        self.padding_factor = padding_factor

    def sample_down(self, factor, fresnelNumber):
        ## TODO this does only work is the image is devided by the factor.
        self.projs_active = self.projs.reshape(
            -1,
            self.projs.shape[1] // factor,
            factor,
            self.projs.shape[2] // factor,
            factor,
        ).mean((2, 4))

        # downsample the masks as 1 is background and 0 is object, we take the min to make the object only bigger in border pixels
        self.masks_active = self.masks.reshape(
            -1,
            self.masks.shape[1] // factor,
            factor,
            self.masks.shape[2] // factor,
            factor,
        ).amin(dim=(2, 4))

        data = dict(
            nVoxels=max(int(1/fresnelNumber), self.projs_active.shape[1] + 50) ** 3,
            diameter=max(int(1/fresnelNumber), self.projs_active.shape[1] + 50),
            accuracy=0.5,
            filter="ram-lak",
        )
        self.downsample = factor
        self.data = data

        self.geo = ParallelGeometry(data)
        self.near, self.far = self.get_near_far(self.geo)
        rays = self.get_rays( factor, self.geo, self.shifts)
        self.rays = torch.cat(
            [
                rays,
                torch.ones_like(rays[..., :1]) * self.near,
                torch.ones_like(rays[..., :1]) * self.far,
            ],
            dim=-1,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        rays = self.rays[index].reshape(-1, 8)
        projs = self.projs_active[index]
        mask = self.masks_active[index]
        out = {
            "projs": projs.to(dtype=torch.float32),
            "rays": rays.to(dtype=torch.float32),
            "idx": index,
            "projs_mask": mask,
        }
        return out

    def get_rays(self, downsampling, geo: ParallelGeometry, shifts):
        """
        Get rays given angles and x-ray machine geometry.
        """

        W, H = geo.nDetector
        rays = []

        for i, angle in enumerate(self.angles):
            pose: torch.Tensor = torch.Tensor(
                self.angle2pose(geo.DSO, angle, shifts[i])
            )
            rays_o, rays_d = None, None
            i, j = torch.meshgrid(
                torch.linspace(0, W - 1, W),
                torch.linspace(0, H - 1, H),
                indexing="ij",
            )  # pytorch"s meshgrid has indexing="ij"
            uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
            vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
            dirs = torch.stack(
                [torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)],
                -1,
            )
            rays_d = torch.sum(
                torch.matmul(pose[:3, :3], dirs[..., None]), -1
            )  # pose[:3, :3] *
            rays_o = torch.sum(
                torch.matmul(
                    pose[:3, :3],
                    torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None],
                ),
                -1,
            ) + pose[:3, -1].expand(rays_d.shape)
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle, shifts):
        R1 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0, 1],
                [0.0, -1, 0],
            ]
        )
        R2 = np.array(
            [
                [0, -1, 0.0],
                [1, 0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R3 = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        shift = rot @ np.array([shifts[0]/self.downsample, shifts[1]/self.downsample, 0])
       # shift = np.array([shifts[0] * np.cos(angle), shifts[1] * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans - shift
        return T

    def get_near_far(self, geo: ParallelGeometry, tolerance=0.005):
        """
        Compute the near and far threshold.
        """
        dist1 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist2 = np.linalg.norm(
            [geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist3 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2]
        )
        dist4 = np.linalg.norm(
            [geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2]
        )
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
