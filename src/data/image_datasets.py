import os
import random
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted
import torchvision.transforms as transforms


def _load_pil_image(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    return pil_image


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,
    model='FRUnet', mode='train'
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = ImageDataset(
        data_dir=data_dir,
        resolution=image_size,
        model=model,
        mode=mode,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(self, data_dir, resolution, model, mode, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.mode = mode

        image_dir = bf.join(data_dir, mode, "images")
        io_dir = bf.join(data_dir, model, mode)

        image_files = natsorted(os.listdir(image_dir))
        input_files = natsorted([f for f in os.listdir(io_dir) if "image" in f])
        target_files = natsorted([f for f in os.listdir(io_dir) if "mask" in f])

        self.local_paths = []
        for img_file, input_file, target_file in zip(image_files, input_files, target_files):
            self.local_paths.append({
                "image": bf.join(image_dir, img_file),
                "input": bf.join(io_dir, input_file),
                "target": bf.join(io_dir, target_file),
            })

        self.local_paths = self.local_paths[shard::num_shards]

    def __len__(self):
        return len(self.local_paths)

    def __getitem__(self, idx):
        paths = self.local_paths[idx]

        pil_image = _load_pil_image(paths["image"]).convert("RGB")
        pil_input = _load_pil_image(paths["input"]).convert("L")
        pil_target = _load_pil_image(paths["target"]).convert("L")

        resize = transforms.Resize((self.resolution, self.resolution), interpolation=Image.BICUBIC)
        pil_image = resize(pil_image)
        pil_input = resize(pil_input)
        pil_target = resize(pil_target)

        # train transforms
        if self.mode == 'train':
            # get random parameters
            # limit rotations to multiples of 90 degrees to avoid interpolation artifacts
            angle = random.choice([0, 90, 180, 270])
            do_vflip = random.random() > 0.5
            do_hflip = random.random() > 0.5

            def apply_common_transforms(img):
                img = transforms.functional.rotate(img, angle, interpolation=transforms.InterpolationMode.BICUBIC)
                if do_vflip:
                    img = transforms.functional.vflip(img)
                if do_hflip:
                    img = transforms.functional.hflip(img)
                return img

            pil_image = apply_common_transforms(pil_image)
            pil_input = apply_common_transforms(pil_input)
            pil_target = apply_common_transforms(pil_target)

        arr_target = np.array(pil_target)
        arr_input = np.array(pil_input)
        arr_image = np.array(pil_image)

        arr_target = np.expand_dims(arr_target, axis=0)
        arr_input = np.expand_dims(arr_input, axis=0)
        arr_image = np.transpose(arr_image, [2, 0, 1])

        # Normalize to float32 in range [0, 1]
        arr_target = arr_target.astype(np.float32) / 255.0
        arr_input = arr_input.astype(np.float32) / 255.0
        arr_image = arr_image.astype(np.float32) / 255.0

        return arr_target, arr_input, arr_image


if __name__ == '__main__':
    print("Running a simple test for load_data...")

    data_dir = 'data/OCTA500_3M'
    batch_size = 2
    image_size = 256

    try:
        data_loader = load_data(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            deterministic=True
        )

        inputs, targets, images = next(data_loader)

        print(f"\nSuccessfully loaded one batch of data.")
        print(f"Batch size: {batch_size}")
        print(f"Image batch shape: {images.shape}")
        print(f"Input batch shape: {inputs.shape}")
        print(f"Target batch shape: {targets.shape}")

        assert images.shape == (batch_size, 3, image_size, image_size)
        assert inputs.shape == (batch_size, 1, image_size, image_size)
        assert targets.shape == (batch_size, 1, image_size, image_size)

        print("\nTest passed: Data shapes are correct.")

    except FileNotFoundError:
        print(f"\n[ERROR] Test failed: Data directory '{data_dir}' not found.")
        print("Please make sure you are running this script from the root of the 'PoissonDiff' repository.")
    except StopIteration:
        print(f"\n[ERROR] Test failed: The data loader is empty.")
        print(f"Check if the directory '{data_dir}/train/images' contains any images.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
