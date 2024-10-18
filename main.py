import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_path", type=str, default=None)

def cli_main():
    torch.set_float32_matmul_precision("medium")

    cli = LightningCLI(
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "ddp",
            "log_every_n_steps": 1,
        }
    )


if __name__ == "__main__":
    cli_main()
