# HoloINR
Reproducibility repository of this [conference paper](https://ieeexplore.ieee.org/abstract/document/11463026) using Implicit Neural Representations (INRs) to solve the coupled inverse problem of holotomography (Holography and Tomography). The code is developed for the near-field holography case, i.e. Fresnel propagation is used to describe the physics between object and detector.


## Getting started
The code was developed under Python 3.11, the required packages are in [requirements.txt](https://github.com/DESY-FS-PETRA/HoloINR/blob/main/requirements.txt)

## Data
Our code expects an h5 file containing the following keys:
- ``angles`` list of floating numbers in $\left[ 0, \dots,  \pi \right)$
- ``data`` the holographic measurement data. First axis corresponds to angles, last two axis are the image dimensions
- ``flat`` flatfield measurements. First axis corresponds to realizations, last two axis are the image dimensions

Make sure the following entries in the [config](https://github.com/DESY-FS-PETRA/HoloINR/blob/main/config/config.yaml) agree to your data/set up:
```
  BeamSetup:
      z01: 61.12e7   # nm
      z02: 19.145e9  # nm
      px_size: 6500  # nm
      energy: 17     # keV
[...]
  Wandb:
      project: "my-project"
      entity: "my-entity"
      mode: "online"        # disabled
[...]
  Dataset:
      data_path: './path/to/data.h5'
      preprocessed_path: "./path/to/preproc/data.h5" # preprocessed data will be saved here
      use_every_nth_angle: 1
      padding_factor: 1
      crop_idcs: [200, 2048, 104, 1952] # x_min, x_max, y_min, y_max

```

This project logs training metrics to Weights & Biases ([wandb](https://wandb.ai/)). To use logging, you’ll need a W&B account. If you prefer not to use W&B, disable it in the settings/config by setting the mode to "disabled".


## Run the Code
After modifying the [config](https://github.com/DESY-FS-PETRA/HoloINR/blob/main/config/config.yaml), please run [trainer.py](https://github.com/DESY-FS-PETRA/HoloINR/blob/main/trainer.py). We ran the code on Nvidia H100 or H200 and expect it to be much slower on older GPUs. 

For initial tests we recommend 
- use a Optimization.downsampling of 4 or 8
- use data sets with only a few angles or set Dataset.use_every_nth_angle > 1 to reduce the amount of data
If you run into memory issues
- decrease RayPropagator.netchunk and RayPropagator.num_grad_samples

In case you encounter any problems please open an issue.

## Citation
Please cite
```
@inproceedings{gruenXRayNearFieldHolotomography2026,
  title = {X-{{Ray Near-Field Holotomography Reconstruction Using Implicit Neural Representations}}},
  booktitle = {{{ICASSP}} 2026 - 2026 {{IEEE International Conference}} on {{Acoustics}}, {{Speech}} and {{Signal Processing}} ({{ICASSP}})},
  author = {Gruen, Johannes and Eberle, Sebastian and Greving, Imke and Flenner, Silja and Burger, Martin and Schroer, Christian G. and Hagemann, Johannes},
  year = 2026,
  month = may,
  pages = {21927--21931},
  publisher = {IEEE},
  address = {Barcelona, Spain},
  doi = {10.1109/ICASSP55912.2026.11463026},
  urldate = {2026-04-27},
  copyright = {https://doi.org/10.15223/policy-029},
  isbn = {979-8-3315-6701-9},
}
```
