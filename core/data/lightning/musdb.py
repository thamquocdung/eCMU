from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import List


from core.data.dataset import FastMUSDB
from core.data.augment import CPUBase


class MUSDB(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        seq_duration: float = 6.0,
        samples_per_track: int = 64,
        random: bool = False,
        random_track_mix: bool = False,
        transforms: List[CPUBase] = None,
        batch_size: int = 16
    ):
        """Initialize MUSDB Loader
        Args:
            root: Data root path
            seq_duration: Duration of each chunk (Note: seq_duration=-1 means loading full song)
            samples_per_track: Number sample chunks for each track 
            random: Random the start time of each chunk or not
            random_track_mix: Randomly mix the component stems from diffent tracks to create new mixture
            transforms: List of Audio Agumentation functions
            batch_size: Batch size
        Returns:
            train_dataset: chunk-level dataset
            val_dataset: full-song dataset
            test_dataset: full-song dataset
        """
        super().__init__()
        self.save_hyperparameters(
            "root",
            "seq_duration",
            "samples_per_track",
            "random",
            "random_track_mix",
            "batch_size",
        )
        # manually save transforms since pedalboard is not pickleable
        if transforms is None:
            self.transforms = None
        else:
            self.transforms = Compose(transforms)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = FastMUSDB(
                root=self.hparams.root,
                subsets=["train"],
                seq_duration=self.hparams.seq_duration,
                samples_per_track=self.hparams.samples_per_track,
                random=self.hparams.random,
                random_track_mix=self.hparams.random_track_mix,
                transform=self.transforms,
                split="train"
            )

        if stage == "validate" or stage == "fit":
            # self.val_dataset = FastMUSDB(
            #     root=self.hparams.root,
            #     subsets=["test"],
            #     seq_duration=self.hparams.seq_duration,
            # )
            self.val_dataset = FastMUSDB(
                root=self.hparams.root,
                subsets=["train"],
                seq_duration=-1, #self.hparams.seq_duration,
                samples_per_track=self.hparams.samples_per_track,
                split="valid"
            )
        if stage == "test":
            self.test_dataset = FastMUSDB(
                root=self.hparams.root,
                subsets=["test"],
                seq_duration=-1,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=1, num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=4
        )
