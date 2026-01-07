import torch


def tv3d(x):
    """Compute isotropic Total Variation for 3D volume"""
    dx = x[1:, :, :] - x[:-1, :, :]
    dy = x[:, 1:, :] - x[:, :-1, :]
    dz = x[:, :, 1:] - x[:, :, :-1]

    tv = torch.sqrt(
        dx[:, :-1, :-1] ** 2 + dy[:-1, :, :-1] ** 2 + dz[:-1, :-1] ** 2 + 1e-6
    )
    return tv.mean()


class RayPropagator:
    def __init__(
        self,
        model,
        n_samples,
        bound,
        netchunk=1024,
        num_grad_samples=1000,
    ):
        self.model = model
        self.n_samples = n_samples
        self.bound = bound
        self.mask = None
        self.downsample = 1
        self.num_grad_samples = num_grad_samples

        self.netchunk = netchunk
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_down(self, downsampling):
        self.active_n_samples = int(self.n_samples / downsampling)
        #   self.netchunk = self.active_n_samples * 10
        # increase netchunk to keep the same number of rays per chunk
        self.downsample = downsampling

    def pts_from_rays(self, rays, idx):
        """
        Get the 3D points from the rays.
        """
        rays_o, rays_d, near, far = (
            rays[..., :3],
            rays[..., 3:6],
            rays[..., 6:7],
            rays[..., 7:],
        )

        # apply learnable offset
        rays_o = rays_o 
        rays_d = rays_d 

        t_vals = torch.linspace(
            0.0, 1.0, steps=self.s, device=self.device
        )
        # as near and far have dims [batch] x [n_rays] x [1] the mul extends the last dim to active_n_samples
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

#        rays_o += (
 #           (torch.rand(rays_o.shape, device=self.device) - 0.5) * self.downsample / 4
  #      )
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        bound = self.bound - 1e-6
        pts = pts.clamp(-bound, bound)

        return pts  # , z_vals, rays_d, rays_o

    def run_network(self, rays, idx):
        """
        Prepares inputs and applies network "fn".
        """
        if rays.shape[0] == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        out_list = []
        for i in range(0, rays.shape[-2], self.netchunk):
            # pts, z_vals, rays_d, rays_o = self.pts_from_rays(rays[i:i + self.netchunk])
            pts = self.pts_from_rays(rays[:, i : i + self.netchunk], idx)
            raw = self.model(
                pts.to(self.device)
            )  # dims: batch x rays/netchunk x active_n_samples (z-direction) x 2

            out = torch.sum(raw, dim=-2)
            out_list.append(out)
        return torch.cat(out_list, dim=1)

    def render(self, rays, idx, mask=None):
        """
        Given a set of rays, compute the output image.
        Therefore, we sample the rays and evaluate the network and sum up the result.
        """
        self.s = int(rays.shape[1] ** 0.5)
        if mask is not None:
            # select rays according to mask
            rays_through_mask = (
                torch.tensor(range(rays.shape[1])).reshape(self.s, self.s).to(self.device)
                * mask.logical_not()
            )
            relevant_idcs = rays_through_mask[rays_through_mask != 0].flatten()
        else:
            relevant_idcs = torch.arange(rays.shape[1])
        num_samples = min(self.num_grad_samples, len(relevant_idcs))
        shuffled_idcs = relevant_idcs[torch.randperm(len(relevant_idcs))]
        gradients = shuffled_idcs[:num_samples]
        rays_gradients = rays[:, gradients]
        acc = self.run_network(rays_gradients, idx)

        acc_out = torch.zeros(
            (*rays.shape[:2], 2), dtype=torch.float32, device=self.device
        )
        acc_out[:, gradients] = acc

        # Process no-gradient rays if there are any
        if num_samples < len(relevant_idcs):
            with torch.no_grad():
                # include rays that were not propagated because of num_samples
                no_grad = shuffled_idcs[num_samples:]

                rays_no_gradients = rays[:, no_grad]
                acc_no_grad = self.run_network(rays_no_gradients, idx)
                acc_out[:, no_grad] = acc_no_grad

        acc = acc_out.reshape(rays.shape[0], self.s, self.s, 2)

        if torch.isnan(acc).any() or torch.isinf(acc).any():
            print("! [Numerical Error] contains nan or inf.")

        return acc

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0.0):
        """Transforms model"s predictions to semantically meaningful values."""
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([1e-10]).expand(dists[:, :1].shape).to(dists.device)],
            -1,
        )
        dists = dists * torch.norm(rays_d[..., :], dim=-1)[:, None]

        acc = torch.sum(raw * dists, dim=-2)
        return acc
