"""
Test script for RectifiedFlow (Poisson version) with real dataset and UNet, running for multiple steps.
"""

import autorootcwd
import matplotlib.pyplot as plt
import torch

from flow.flow import Flow
from src.data.image_datasets import load_data
from src.model.unet import UNetModel
from src.utils.script_util import create_model

# Configurations (adjust as needed)
BATCH_SIZE = 2
IMAGE_SIZE = 64
CHANNELS = 1
DATASET = "OCTA500_6M"
DATA_DIR = "data/OCTA500_6M"
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
STEPS = 10
EPOCHS = 1000


def main():
    # Load dataset (returns DataLoader)
    dataloader = load_data(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        class_cond=False,
        model="Unet",
        mode="train",
        deterministic=True,
    )

    # Create UNet model
    model = create_model(
        IMAGE_SIZE,
        CHANNELS,
        num_channels=128,
        num_res_blocks=2,
        use_checkpoint=False,
        attention_resolutions="16,8",
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
    )
    model = model.to(DEVICE)

    # Create RectifiedFlow
    flow = Flow(model, num_timesteps=STEPS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(EPOCHS):
        for batch in dataloader:
            x_0, x_1, image = batch  # input, target, image
            x_0 = x_0.to(DEVICE)
            x_1 = x_1.to(DEVICE)
            model.train()
            optimizer.zero_grad()
            loss_dict = flow.training_losses(model, x_0, x_1)
            loss = loss_dict["loss"].mean()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            break  # Remove this break to train on full epoch

    # Sampling test
    model.eval()
    with torch.no_grad():
        x_0 = torch.poisson(x_0)
        x_sampled = flow.sample(x_0)
        print("Sampled x shape:", x_sampled.shape)
        plt.imsave(
            "sampled_image.png",
            x_sampled[0, 0].cpu().numpy(),
            cmap="gray",
        )


if __name__ == "__main__":
    main()
