from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl


from src.Blocks import DownBlock
from src.Blocks import BottleNeck
from src.Blocks import UpBlock

import numpy as np

from torch.utils.data import DataLoader

import os

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision('medium')


class Runet(nn.Module):
    def __init__(self, **kwargs):
        super(Runet, self).__init__(**kwargs)

        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding="same", stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.down2 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DownBlock(64, 64, r=False),
            DownBlock(64, 64, r=False),
            DownBlock(64, 64, r=False),
            DownBlock(64, 128, r=True))
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2),
            DownBlock(128, 128, r=False),
            DownBlock(128, 128, r=False),
            DownBlock(128, 128, r=False),
            DownBlock(128, 256, r=True))

        self.down4 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DownBlock(256, 256, r=False),
            DownBlock(256, 256, r=False),
            DownBlock(256, 256, r=False),
            DownBlock(256, 256, r=False),
            DownBlock(256, 256, r=False),
            DownBlock(256, 512, r=True))

        self.down5 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            DownBlock(512, 512, r=False),
            DownBlock(512, 512, r=False),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.BottleNeck = nn.Sequential(
            BottleNeck(512, 1024),
            BottleNeck(1024, 512))

        self.up1 = UpBlock(512, 512, 1024, scale=1)
        self.up2 = UpBlock(512, 384, 640, scale=2)
        self.up3 = UpBlock(384, 256, 352, scale=2)
        self.up4 = UpBlock(256, 96, 192, scale=2)

        self.up5 = UpBlock(96, 99, 88, scale=2, last=False)
        self.conv = nn.Conv2d(99, 3, kernel_size=1, padding="same", stride=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x6 = self.BottleNeck(x5)

        x7 = self.up1(x6, x5)
        x8 = self.up2(x7, x4)
        x9 = self.up3(x8, x3)
        x10 = self.up4(x9, x2)
        x11 = self.up5(x10, x1)
        return self.conv(x11)


class RunetModel(pl.LightningModule):
    def __init__(self, tr_dataset, vl_dataset, ts_dataset, lr=1e-4,save_every_n_epoch=100, **kwargs):
        super(RunetModel, self).__init__(**kwargs)
        self.lr = lr
        self.save_hyperparameters()
        self.tr_dataset = tr_dataset
        self.tr_dataset_loader = None
        self.vl_dataset = vl_dataset
        self.vl_dataset_loader = None
        self.ts_dataset = ts_dataset
        self.ts_dataset_loader = None
        self.save_every_n_epoch = save_every_n_epoch
        self.loss = nn.MSELoss()
        self.model = Runet().to(dev)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        print(loss)
        return loss

    def train_dataloader(self):
        if self.tr_dataset_loader is None:
            self.tr_dataset_loader = DataLoader(
                self.tr_dataset, batch_size=32, shuffle=False, num_workers=0)

        return self.tr_dataset_loader

    def val_dataloader(self):
        if self.vl_dataset_loader is None:
            self.vl_dataset_loader = DataLoader(
                self.vl_dataset, batch_size=32, shuffle=False, num_workers=0)
        return self.vl_dataset_loader

    def test_dataloader(self):
        if self.ts_dataset_loader is None:
            self.ts_dataset_loader = DataLoader(
                self.ts_dataset, batch_size=32, shuffle=False, num_workers=0)
        return self.ts_dataset_loader

    def on_train_epoch_end(self):
        ts_dataloader = self.train_dataloader()
        clear_output(wait=True)
        if not os.path.exists("images"):
            os.mkdir("images")
        for i, (img1, img2) in enumerate(ts_dataloader):
            img1 = img1.to(dev)
            img2 = img2.to(dev)
            img1_hat = self(img1)
            # put images in renge 0 to 255 numpy
            img1 = np.transpose(img1[0].cpu().numpy(), (1, 2, 0))
            img1 = np.clip(img1, 0, 1)
            img2 = np.transpose(img2[0].cpu().numpy(), (1, 2, 0))
            img2 = np.clip(img2, 0, 1)
            img1_hat = np.transpose(img1_hat[0].cpu().detach().numpy(), (1, 2, 0))
            img1_hat = np.clip(img1_hat, 0, 1)

            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(img1)
            plt.subplot(1, 3, 2)
            plt.imshow(img2)
            plt.subplot(1, 3, 3)
            plt.imshow(img1_hat)
            plt.savefig(f"images/{self.current_epoch}_{i}.png")
            plt.show()
            if i == 2:
                break
        if self.current_epoch % self.save_every_n_epoch == 0:
            self.save_checkpoint()

    def save_checkpoint(self ,path= None ):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        if path is None:
            path = f"checkpoints/{self.current_epoch}.pth"
            
        torch.save(self.state_dict(), path)
        print(f"Checkpoint saved at {self.current_epoch}")

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
        print("Checkpoint loaded")
