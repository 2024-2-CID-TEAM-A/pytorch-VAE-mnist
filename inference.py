import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn.functional as F
from torchvision.utils import save_image

CUDA = True
DATA_PATH = "./data"
BATCH_SIZE = 48
IMAGE_CHANNEL = 1
INITIAL_CHANNEL = 4
Z_DIM = 16
seed = 1

CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda" if CUDA else "cpu")
cudnn.benchmark = True

dataset = dset.MNIST(
    root=DATA_PATH,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

VAEdataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1).to(device)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], 8, 7, 7).to(device)


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=IMAGE_CHANNEL,
        output_channels=INITIAL_CHANNEL,
        z_dim=Z_DIM,
    ):
        super(VAE, self).__init__()
        h_dim = 8 * 7 * 7

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
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim).to(device)
        self.fc2 = nn.Linear(h_dim, z_dim).to(device)
        self.fc3 = nn.Linear(z_dim, h_dim).to(device)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(
                8, 4, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        logvar = torch.clamp(logvar, min=-4, max=4)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar


model = torch.load("./checkpoint/vae_mnist_28x28.pth", map_location=device)

with torch.no_grad():
    images, label = next(iter(VAEdataloader))
    images = images.float().to(device)

    recon, _, _ = model(images)

    noise = torch.randn_like(images).to(device)
    generated, _, _ = model(noise)

    save_gt_path = "./ground_truth"
    os.makedirs(save_gt_path, exist_ok=True)
    save_generated_path = "./generated"
    os.makedirs(save_generated_path, exist_ok=True)
    save_recon_path = "./recon"
    os.makedirs(save_recon_path, exist_ok=True)

    save_image(images, os.path.join(save_gt_path, "vae_mnist_28x28.png"))
    save_image(generated, os.path.join(save_generated_path, "vae_mnist_28x28.png"))
    save_image(recon, os.path.join(save_recon_path, "vae_mnist_28x28.png"))
