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
Z_DIM = 16
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
        # h_dim = 8 * 7 * 7 = 392
        return input.view(input.size()[0], 8, 7, 7).to(device)


class VAE(nn.Module):
    def __init__(
        self,
        image_channels=IMAGE_CHANNEL,
        output_channels=INITIAL_CHANNEL,
        z_dim=Z_DIM,
    ):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        h_dim = 8 * 7 * 7

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

        self.fc1 = nn.Linear(h_dim, z_dim).to(device)
        self.fc2 = nn.Linear(h_dim, z_dim).to(device)
        self.fc3 = nn.Linear(z_dim, h_dim).to(device)

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
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + 3 * KLD, MSE, 3 * KLD


from torch.utils.tensorboard import SummaryWriter

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

writer = SummaryWriter("runs/mnist_28x28")
epochs = 500

for epoch in range(epochs):
    for idx, (images, label) in enumerate(
        tqdm(VAEdataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    ):
        optimizer.zero_grad()
        images = images.float().to(device)

        recon_images, mu, logvar = model(images)
        loss, mse, kld = loss_fn(recon_images, images, mu, logvar)

        if torch.isnan(loss).any():
            print("NaN value in loss!")
            break

        loss.backward()
        optimizer.step()

    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, MSE: {mse.item():.4f}, KLD: {abs(kld.item()):.4f}"
    )
    print(
        f"Mu range: {torch.min(mu[0])} ~ {torch.max(mu[0])}, Logvar range: {torch.min(logvar[0])} ~ {torch.max(logvar[0])}"
    )

    writer.add_scalar("Training Loss", loss.item(), epoch)

writer.close()

if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")
torch.save(model, "./checkpoint/vae_mnist_28x28.pth")
