import io
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from PIL import Image, ImageChops, ImageEnhance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# -----------------------------
# PATH SETTINGS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ela_cnn_model.pth")

IMAGE_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_dataset_dir():
    candidates = [
        os.path.join(BASE_DIR, "real-vs-fake"),
        os.path.join(BASE_DIR, "real_vs_fake"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    checked_paths = ", ".join(candidates)
    raise FileNotFoundError(f"Dataset folder not found. Checked: {checked_paths}")


DATASET_DIR = resolve_dataset_dir()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
tensor_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# -----------------------------
# ELA IMAGE FUNCTION
# -----------------------------
def convert_pil_to_ela_image(image, quality=90):
    original = image.convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        compressed = compressed.convert("RGB")
        ela = ImageChops.difference(original, compressed)

    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    ela = ela.convert("RGB")

    return original, ela


def convert_to_ela_image(image_path, quality=90):
    with Image.open(image_path) as source:
        return convert_pil_to_ela_image(source, quality=quality)


def prepare_ela_tensor(image):
    _, ela_image = convert_pil_to_ela_image(image)
    return tensor_transform(ela_image)


# -----------------------------
# CNN MODEL
# -----------------------------
class ELACNN(nn.Module):

    def __init__(self):
        super(ELACNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    print("\nTraining Model...")

    train_dataset = datasets.ImageFolder(
        os.path.join(DATASET_DIR, "train"),
        transform=prepare_ela_tensor
    )

    # fast training subset
    if len(train_dataset) > 6000:
        indices = random.sample(range(len(train_dataset)), 6000)
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ELACNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10  # increased for better training graph
    epoch_losses = []
    batch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())
            batch_count += 1

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("\nModel Saved:", MODEL_PATH)

    return epoch_losses, batch_losses


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = ELACNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# -----------------------------
# PREDICT IMAGE
# -----------------------------
def predict_image(image_path):
    model = load_model()
    original, ela_img = convert_to_ela_image(image_path)
    tensor = tensor_transform(ela_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, 1)

    confidence = probs[0][pred.item()].item() * 100
    label = "REAL" if pred.item() == 1 else "FAKE"
    return original, ela_img, label, confidence


# -----------------------------
# COLLECT 4 TEST SAMPLES (2 real + 2 fake)
# -----------------------------
def collect_test_samples(num_per_class=2):
    """Collect `num_per_class` real and fake images from the test set."""
    test_dir = os.path.join(DATASET_DIR, "test")

    real_images = []
    fake_images = []

    for root, dirs, files in os.walk(test_dir):
        folder_name = os.path.basename(root).lower()
        is_real = "real" in folder_name
        is_fake = "fake" in folder_name or "ai" in folder_name or "forged" in folder_name

        for file in sorted(files):
            if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_path = os.path.join(root, file)
                if is_real and len(real_images) < num_per_class:
                    real_images.append(image_path)
                elif is_fake and len(fake_images) < num_per_class:
                    fake_images.append(image_path)

        if len(real_images) >= num_per_class and len(fake_images) >= num_per_class:
            break

    # Fallback: if folder names don't match, just grab any 4 images
    all_images = real_images + fake_images
    if len(all_images) < 4:
        print("Warning: Could not find separate real/fake folders. Grabbing first 4 images.")
        all_images = []
        for root, dirs, files in os.walk(test_dir):
            for file in sorted(files):
                if file.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    all_images.append(os.path.join(root, file))
                    if len(all_images) >= 4:
                        break
            if len(all_images) >= 4:
                break

    return all_images[:4]


# -----------------------------
# PLOT TRAINING GRAPH
# -----------------------------
def plot_training_graph(epoch_losses, batch_losses):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Epoch-level loss
    epochs = list(range(1, len(epoch_losses) + 1))
    axes[0].plot(epochs, epoch_losses, color="#58a6ff", linewidth=2.5,
                 marker="o", markersize=6, markerfacecolor="#f78166", markeredgecolor="white", markeredgewidth=1.5)
    axes[0].fill_between(epochs, epoch_losses, alpha=0.15, color="#58a6ff")
    axes[0].set_title("Training Loss per Epoch", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Average Loss", fontsize=11)
    axes[0].set_xticks(epochs)
    axes[0].grid(True, linestyle="--", alpha=0.3, color="#8b949e")

    # Annotate min loss
    min_idx = epoch_losses.index(min(epoch_losses))
    axes[0].annotate(f"Min: {epoch_losses[min_idx]:.4f}",
                     xy=(epochs[min_idx], epoch_losses[min_idx]),
                     xytext=(epochs[min_idx] + 0.3, epoch_losses[min_idx] + 0.02),
                     fontsize=9, color="#3fb950",
                     arrowprops=dict(arrowstyle="->", color="#3fb950"))

    # Batch-level loss
    batch_x = list(range(1, len(batch_losses) + 1))
    axes[1].plot(batch_x, batch_losses, color="#bc8cff", linewidth=0.8, alpha=0.7)
    # Smoothed line
    window = max(1, len(batch_losses) // 30)
    smoothed = np.convolve(batch_losses, np.ones(window) / window, mode="valid")
    axes[1].plot(range(window, len(batch_losses) + 1), smoothed,
                 color="#f78166", linewidth=2.2, label="Smoothed")
    axes[1].set_title("Training Loss per Batch", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Batch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.3, color="#8b949e")
    axes[1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    fig.suptitle("ELA-CNN Training Progress", fontsize=15, fontweight="bold",
                 color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "training_graph.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("Training graph saved: training_graph.png")


# -----------------------------
# DISPLAY 4 IMAGES WITH PREDICTIONS
# -----------------------------
def show_four_results(image_paths):
    """Display 4 images: original + ELA + prediction label in a clean grid."""
    n = len(image_paths)
    if n == 0:
        print("No test images found.")
        return

    # Figure: 3 rows (original, ELA, label bar) × n cols
    fig = plt.figure(figsize=(5 * n, 13))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(3, n, figure=fig, hspace=0.08, wspace=0.04,
                           height_ratios=[5, 5, 0.7])

    for col, image_path in enumerate(image_paths):
        try:
            original, ela_img, label, confidence = predict_image(image_path)
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            continue

        is_real = label == "REAL"
        accent_color = "#3fb950" if is_real else "#f78166"   # green / red-orange
        bg_color     = "#0a1f0a" if is_real else "#1f0a0a"

        # --- Row 0: Original Image ---
        ax_orig = fig.add_subplot(gs[0, col])
        ax_orig.imshow(original)
        ax_orig.set_title("Original Image", fontsize=10, color="#c9d1d9",
                          fontweight="bold", pad=6)
        ax_orig.axis("off")
        # coloured border
        for spine in ax_orig.spines.values():
            spine.set_edgecolor(accent_color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # --- Row 1: ELA Image ---
        ax_ela = fig.add_subplot(gs[1, col])
        ax_ela.imshow(ela_img)
        ax_ela.set_title("ELA Image", fontsize=10, color="#c9d1d9",
                         fontweight="bold", pad=6)
        ax_ela.axis("off")
        for spine in ax_ela.spines.values():
            spine.set_edgecolor(accent_color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # --- Row 2: Prediction label bar ---
        ax_label = fig.add_subplot(gs[2, col])
        ax_label.set_facecolor(bg_color)
        icon = "✅" if is_real else "❌"
        ax_label.text(
            0.5, 0.5,
            f"{icon}  {label}  |  {confidence:.1f}%",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=accent_color,
            transform=ax_label.transAxes
        )
        ax_label.axis("off")
        for spine in ax_label.spines.values():
            spine.set_edgecolor(accent_color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    fig.suptitle("ELA-CNN  ·  Fake Image Detection Results",
                 fontsize=16, fontweight="bold", color="white",
                 y=0.98)

    plt.savefig(os.path.join(BASE_DIR, "detection_results.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print("Detection results saved: detection_results.png")


# -----------------------------
# MAIN PROGRAM
# -----------------------------
if __name__ == "__main__":

    print("=" * 45)
    print("      ELA CNN FAKE IMAGE DETECTOR")
    print("=" * 45)
    print(f"Device : {DEVICE}")
    print(f"Dataset: {DATASET_DIR}")

    # ---------- TRAIN ----------
    epoch_losses, batch_losses = train_model()

    # ---------- TRAINING GRAPH ----------
    print("\nGenerating training graph...")
    plot_training_graph(epoch_losses, batch_losses)

    # ---------- COLLECT 4 TEST SAMPLES ----------
    print("\nCollecting test samples (2 real + 2 fake)...")
    image_paths = collect_test_samples(num_per_class=2)

    if not image_paths:
        print("No test images found — please check your dataset structure.")
    else:
        print(f"Found {len(image_paths)} test images.")
        for p in image_paths:
            print("  ", p)

        # ---------- SHOW 4-IMAGE DETECTION GRID ----------
        print("\nRunning detection on 4 images...")
        show_four_results(image_paths)

    print("\nProgram Completed ✓")