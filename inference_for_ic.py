import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np

CUDA = True
DATA_PATH = "./data"
BATCH_SIZE = 48
GENERATED_IMGS_PATH = "./generated_imgs"
NUM_IMAGES = 500
IMAGE_CHANNEL = 1
Z_DIM = 16

device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(GENERATED_IMGS_PATH, exist_ok=True)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1).to(device)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 8, 7, 7).to(device)


class VAE(nn.Module):
    def __init__(self, image_channels=IMAGE_CHANNEL, output_channels=4, z_dim=Z_DIM):
        super(VAE, self).__init__()
        h_dim = 8 * 7 * 7  # 392

        self.encoder = nn.Sequential(
            nn.Conv2d(
                image_channels, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(
                output_channels, output_channels * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)  # mu
        self.fc2 = nn.Linear(h_dim, z_dim)  # logvar
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp_()
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def encode_bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        logvar = torch.clamp(logvar, min=-4, max=4)
        z = self.reparameterize(mu, logvar)
        return z

    def encode(self, x):
        h = self.encoder(x)
        return self.encode_bottleneck(h)

    def decode(self, z):
        z = F.relu(self.fc3(z))
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


model_path = "./checkpoint/vae_mnist_28x28.pth"
assert os.path.exists(model_path), f"Model file not found: {model_path}"
model = torch.load(model_path, map_location=device)
model.eval()

print(f"Generating {NUM_IMAGES} images...")
with torch.no_grad():
    num_batches = int(np.ceil(NUM_IMAGES / BATCH_SIZE))
    for batch_idx in range(num_batches):
        batch_size = (
            BATCH_SIZE
            if (batch_idx < num_batches - 1 or NUM_IMAGES % BATCH_SIZE == 0)
            else NUM_IMAGES % BATCH_SIZE
        )
        noise = torch.randn(batch_size, Z_DIM).to(device)
        generated_images = model.decode(noise).cpu()

        for i, img in enumerate(generated_images):
            img_path = os.path.join(
                GENERATED_IMGS_PATH, f"generated_{batch_idx * BATCH_SIZE + i + 1}.png"
            )
            save_image(img, img_path)

print(f"Images saved to {GENERATED_IMGS_PATH}")
