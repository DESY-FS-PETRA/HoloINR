import logging


class ConeBeam:
    def __init__(self, energy, px_size, z02, z01):
        self.energy = energy
        self._px_size = px_size
        self.active_px_size = px_size
        self.z02 = z02
        self.z01 = z01

    @property
    def px_size(self):
        return self._px_size

    @px_size.setter
    def px_size(self, downsample_factor):
        self.active_px_size = self.px_size * downsample_factor

    def z12(self):
        return self.z02 - self.z01

    def get_fr(self):
        z12 = self.z12()

        lam = 1.2398 / self.energy
        M = (z12 + self.z01) / self.z01
        dx_eff = self.active_px_size / M
        z_eff = z12 / M
        fr_eff = dx_eff**2 / lam / z_eff

        logging.info(f"{'Energy':<17}{self.energy}")
        logging.info(f"{'Lambda':<17}{round(lam, 6)}")
        logging.info(f"{'Magnification':<17}{round(M, 2)}")
        logging.info(f"{'Effective dx':<17}{int(dx_eff)}")
        logging.info(f"{'Effective z12':<17}{int(z_eff)}")
        logging.info(f"{'Fresnel Number':<17}{fr_eff}")

        return fr_eff
