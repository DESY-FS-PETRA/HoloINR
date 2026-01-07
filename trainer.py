import torch
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
import sys
import os
from pytorch_msssim import ssim
from propagator import Propagator

sys.path.append(".")
from encoder import get_encoder
from network import DensityNetwork
from cone_beam import ConeBeam
from rays import RayPropagator
from fresnel_propagator_torch import FresnelPropagator
from tigre import TIGREDataset
from torch.utils.data import DataLoader
import wandb
import torch.nn.functional as F
from visualize import Logger

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
image_saver_interval = 50


def train(cfg: DictConfig):
    # Start a new wandb run to track this script.
    run = wandb.init(
        project=cfg["Wandb"]["project"],
        entity=cfg["Wandb"]["entity"],
        mode=cfg["Wandb"]["mode"],
        save_code=True,
        notes=cfg.get("Key", "no_key"),
    )
    torch.autograd.set_detect_anomaly(True)
    basepath = "../../.."
    code_artifact = wandb.Artifact(type="code", name="code")
    files = [x for x in os.listdir(basepath) if x.endswith(".py")]
    [code_artifact.add_file(f"{basepath}/{x}") for x in files]
    code_artifact.add_file(f"{basepath}/config/config.yaml")  # log config as well
    wandb.log_artifact(code_artifact)

    # get data
    dataset = TIGREDataset(**cfg["Dataset"])
    dataloader = DataLoader(
        dataset, num_workers=16, batch_size=1, pin_memory=True, shuffle=True
    )
    measurement_dims = dataset.projs.shape[-2:]

    cone_beam = ConeBeam(**cfg["BeamSetup"])
    ## Init INR for 3D object
    encoder = get_encoder(**cfg["Encoder"])
    net = DensityNetwork(
        encoder=encoder,
        last_activation=lambda x: x**2,
        bound=measurement_dims[0],
        **cfg["Network"],
    )
    offset_vectors = torch.nn.Parameter(
        torch.ones((dataset.n_samples), device=device)
    )

    rayPropagator = RayPropagator(
        net, bound=measurement_dims[0], **cfg["RayPropagator"]
    )
    # Initialize model
    # Initialize optimizer outside the loop
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["Optimization"][0]["lr"])
    o_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, cooldown=300, patience=300, factor=0.2
    )

    offsetOptimizer = torch.optim.SGD(
        [offset_vectors], lr=cfg["OffsetOptimization"]["lr"]
    )
    offsetScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        offsetOptimizer, cooldown=2, patience=2
    )
    logger = Logger(run)
    stop = False
    net.train()
    l = 0
    for step in cfg["Optimization"]:
        downsampling = step["downsampling"]
        cone_beam.px_size = downsampling
        dataset.sample_down(downsampling, cone_beam.get_fr())
        net.bound = dataset.data["diameter"]
        rayPropagator.bound = dataset.data["diameter"]
        rayPropagator.sample_down(downsampling)
        
        fresnelPropagator = FresnelPropagator(
            cone_beam.get_fr(), dataset.projs.shape[-2], downsampling, device
        )
        propagator = Propagator(
            fresnelPropagator,
            logger,
            dataset.probe,
            n_angles=dataset.n_samples,
            rayPropagator=rayPropagator,
            downsampling_factor=downsampling,
        )
        propagator.downsampling_factor = downsampling
        for epoch in range(step["epochs"]):

            for batch_idx, data in tqdm(
                enumerate(iterable=dataloader), total=len(dataloader)
            ):
                l+=1
                if l == 4000:
                    stop = True
                    break
                optimizer.zero_grad()
                projs = data["projs"].to(device)
                rays = data["rays"]
                idx = data["idx"]
                projs = torch.sqrt(projs)
                y_processed = propagator(
                    rays.to(device),
                    idx,
                    mask=None,
                    log=(batch_idx % image_saver_interval == 0),
                    offset=offset_vectors[idx].item(),
                )
                ssim_value = ssim(
                    y_processed.unsqueeze(0).unsqueeze(0),
                    projs.unsqueeze(0).unsqueeze(0),
                    data_range=max(y_processed.max(), projs.max())
                    - min(y_processed.min(), projs.min()),
                    size_average=True,
                ).item()

                loss = (
                    F.mse_loss(y_processed, projs)
                    + step["weight_l1_imag"] * propagator.loss.get("L1_imag", 0)
                    + step["weight_l1_real"] * propagator.loss.get("L1_real", 0)
                    + step["weight_tv_imag"] * propagator.loss["TV_imag"]
                    + step["weight_tv_real"] * propagator.loss["TV_real"]
                )
               
                loss.backward()
                optimizer.step()
                o_scheduler.step(F.mse_loss(y_processed, projs).detach())
                logger.log_epoch["loss"] = logger.log_epoch.get("loss", 0) + loss.item()
                if "angular_loss" not in logger.log_epoch:
                    logger.log_epoch["angular_loss"] = dict()
                if "angular_data_fidelity" not in logger.log_epoch:
                    logger.log_epoch["angular_data_fidelity"] = dict()
                if "angular_offset" not in logger.log_epoch:
                    logger.log_epoch["angular_offset"] = dict()
                if "angular_offset_direction" not in logger.log_epoch:
                    logger.log_epoch["angular_offset_direction"] = dict()

                logger.log_epoch["angular_data_fidelity"][idx] = F.mse_loss(
                    y_processed, projs
                ).item()
                logger.log_epoch["angular_loss"][idx] = loss.item()
                logger.log_epoch["angular_offset"][idx] = offset_vectors[idx].item()
                logger.log["loss"] = loss.item()
                logger.log["SSIM"] = ssim_value
                logger.log["optimizer_lr"] = optimizer.param_groups[0]["lr"]
                if logger.log["optimizer_lr"] < 1e-7:
                    stop = True
                    break
                logger.log["data_fidelity"] = F.mse_loss(y_processed, projs).item()
                logger.log["data_fidelity l1"] = F.l1_loss(y_processed, projs).item()
                if batch_idx % image_saver_interval == 0:
                    # Log the images to wandb (only first of the batch)
                    logger.image_from_tensor(projs[0], name="Measurement")
                    logger.image_from_tensor(
                        y_processed[0] - projs[0], name="Error in Hologram Space"
                    )
                    logger.image_from_tensor(
                        (y_processed[0] - projs[0]).abs().log(),
                        name="Abs-Error in Hologram Space - Log Scale",
                    )
                    logger.flush()

            offsetOptimizer.step()
            offsetScheduler.step(torch.tensor(logger.log_epoch["loss"]))
            logger.log["offset_lr"] = offsetOptimizer.param_groups[0]["lr"]

            logger.slice_from_object(net)

            if stop:
                break

    logger.save_h5(net, propagator, step["downsampling"], no_angles=dataset.n_samples)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
