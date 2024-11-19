import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

CUDA = True
DATA_PATH = "./data"
BATCH_SIZE = 512
IMAGE_CHANNEL = 1
INITIAL_CHANNEL = 4
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 5
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 2e-4
seed = 1

CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda" if CUDA else "cpu")
cudnn.benchmark = True
# device = torch.device('cuda')

# Data preprocessing
dataset = dset.MNIST(
    root=DATA_PATH,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(X_DIM),  # Reszie from 28x28 to 64x64
            transforms.ToTensor(),
        ]
    ),
)

# Dataloader
VAEdataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)


class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.size()[0], -1).to(
            device
        )  # for connecting conv layer and linear layer


class UnFlatten(nn.Module):
    def forward(self, input):

        return input.view(input.size()[0], 64, 2, 2).to(
            device
        )  # for connecting linear layer and conv layer


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=IMAGE_CHANNEL,
        output_channels=INITIAL_CHANNEL,
        h_dim=256,
        z_dim=16,
    ):  # h_dim : last hidden dimension, z_dim : latent dimension
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(
                image_channels, output_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(
                output_channels, output_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(),
            nn.Conv2d(
                output_channels * 2,
                output_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels * 4),
            nn.ReLU(),
            nn.Conv2d(
                output_channels * 4,
                output_channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels * 8),
            nn.ReLU(),
            nn.Conv2d(
                output_channels * 8,
                output_channels * 16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(output_channels * 16),
            nn.ReLU(),
            nn.Dropout(0.8),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim).to(
            device
        )  # for mu right before reparameterization
        self.fc2 = nn.Linear(h_dim, z_dim).to(
            device
        )  # for logvar right before reparameterization

        self.fc3 = nn.Linear(z_dim, h_dim).to(device)  # right before decoding starts

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(
                output_channels * 16,
                output_channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(output_channels * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                output_channels * 8,
                output_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(output_channels * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                output_channels * 4,
                output_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                output_channels * 2,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                output_channels,
                image_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(image_channels),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp  # N(mu, std) ~ N(0, 1) * std + mu
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        logvar = torch.clamp(logvar, min=-4, max=4)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.bottleneck(h)

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + 10 * KLD, MSE, 10 * KLD


from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, LinearLR

model = VAE()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/500epochs_KLD*30_BCE")
epochs = 500

for epoch in range(epochs):
    for idx, (images, label) in enumerate(
        tqdm(VAEdataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    ):
        optimizer.zero_grad()

        images = images.float().to(device)  # train image (BATCH_SIZE, 1, 64, 64)
        recon_images, mu, logvar = model(images)

        loss, mse, kld = loss_fn(recon_images, images, mu, logvar)

        if torch.isnan(loss).any():
            print("NaN value in loss!")
            break

        loss.backward()
        optimizer.step()

    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f} (total loss), {mse.item():.4f} (mse), {abs(kld.item()):.4f} (kld)"
    )
    print(
        f"Mu range: {torch.min(mu[0])} ~ {torch.max(mu[0])}, Logvar range: {torch.min(logvar[0])} ~ {torch.max(logvar[0])}"
    )

    writer.add_scalar("Training Loss", loss.item(), epoch)

writer.close()

if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")
torch.save(model, "./checkpoint/500epochs_KLD*30_BCE.pth")
