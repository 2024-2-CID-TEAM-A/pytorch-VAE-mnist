import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
from tqdm import tqdm


def load_inception_model(device):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    return model


def preprocess_for_inception(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),  # Resize for Inception v3
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def get_inception_probs(model, image_tensors, device):
    with torch.no_grad():
        logits = model(image_tensors.to(device))
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs


def calculate_inception_score(image_dir, model, device, splits=10):
    image_paths = [
        os.path.join(image_dir, img)
        for img in os.listdir(image_dir)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"Found {len(image_paths)} images in {image_dir}")

    if len(image_paths) == 0:
        raise ValueError("No images found in the directory.")

    all_preds = []
    for image_path in tqdm(image_paths, desc="Processing Images"):
        image_tensor = preprocess_for_inception(image_path).unsqueeze(0).to(device)
        probs = get_inception_probs(model, image_tensor, device)
        all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    split_scores = []
    for k in range(splits):
        part = all_preds[
            k * (len(all_preds) // splits) : (k + 1) * (len(all_preds) // splits), :
        ]
        p_y = np.mean(part, axis=0)
        kl_div = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        split_scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == "__main__":
    generated_image_dir = "./generated_imgs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    inception_model = load_inception_model(device)

    mean_is, std_is = calculate_inception_score(
        generated_image_dir, inception_model, device
    )
    print(f"\nInception Score: {mean_is:.4f} ± {std_is:.4f}")
