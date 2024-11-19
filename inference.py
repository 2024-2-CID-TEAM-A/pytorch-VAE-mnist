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
from torchvision.utils import save_image

CUDA = True
DATA_PATH = "./data"
BATCH_SIZE = 48  # different from training batch. This is equal to #(generated images)
IMAGE_CHANNEL = 1
INITIAL_CHANNEL = 4
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 5
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


resize_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),  # Convert to 28x28 again
    ]
)


class Flatten(nn.Module):
    def forward(self, input):
        global BATCH_SIZE
        return input.view(input.size()[0], -1).to(device)


class UnFlatten(nn.Module):
    def forward(self, input):
        global BATCH_SIZE
        return input.view(input.size()[0], 64, 2, 2).to(device)


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=IMAGE_CHANNEL,
        output_channels=INITIAL_CHANNEL,
        h_dim=256,
        z_dim=16,
    ):  # h_dim : hidden dimension, z_dim : latent dimension
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
        z = mu + std * esp
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


model = torch.load("./checkpoint/500epochs_KLD_dynamic.pth")  # saved model path
# note : model architecture is saved as well, so p
# you need VAE() class in this file (or you can make it as a module with separation)

# DataLoader for taking only first batch - very naive way to do this!
with torch.no_grad():

    images, label = next(iter(VAEdataloader))  # only take the first batch

    images = images.float().to(device)
    recon, _, _ = model(images)

    noise = torch.randn_like(images)
    generated, _, _ = model(noise)

    # resize from 64x64 to 28x28 for every image in one batch
    ground_truth_images = torch.stack([resize_transform(image) for image in images])
    generated_images = torch.stack([resize_transform(image) for image in generated])
    recon_images = torch.stack([resize_transform(image) for image in recon])

    save_gt_path = "./ground_truth"
    os.makedirs(save_gt_path, exist_ok=True)

    save_generated_path = "./generated"
    os.makedirs(save_generated_path, exist_ok=True)

    save_recon_path = "./recon"
    os.makedirs(save_recon_path, exist_ok=True)

    save_image(
        ground_truth_images.view(BATCH_SIZE, 1, 28, 28),
        "./ground_truth/500epochs_KLD_dynamic" + ".png",
    )
    save_image(
        generated_images.view(BATCH_SIZE, 1, 28, 28),
        "./generated/500epochs_KLD_dynamic" + ".png",
    )
    save_image(
        recon_images.view(BATCH_SIZE, 1, 28, 28),
        "./recon/500epochs_KLD_dynamic" + ".png",
    )
