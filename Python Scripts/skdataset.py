import random
import torch
import torch.utils.data as Data
import scipy.io as sio
import numpy as np

__all__ = ['loader']

class CoreSymmetryAugment:
    """
    90° rotations, horizontal/vertical flips, and small periodic translations.
    Operates on a Tensor of shape (1, 50, 50) with 0/1 unit‐cell data.
    """
    def __init__(self, max_shift: int = 5):
        self.max_shift = max_shift

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # 1) 90° rotation
        k = random.randint(0, 3)
        img = torch.rot90(img, k, dims=(1, 2))
        # 2) horizontal / vertical flips
        if random.random() < 0.5:
            img = torch.flip(img, dims=(1,))  # horizontal
        if random.random() < 0.5:
            img = torch.flip(img, dims=(2,))  # vertical
        # 3) periodic roll (±max_shift pixels)
        dx = random.randint(-self.max_shift, self.max_shift)
        dy = random.randint(-self.max_shift, self.max_shift)
        img = torch.roll(img, shifts=(dx, dy), dims=(1, 2))
        return img


def loader(num: int,
           subset_size: int = None,
           batch_size: int = 128,
           augment: bool = True):
    """
    Returns:
      test_loader, train_loader
    where each batch is (imgs, labels) and imgs have shape B×1×50×50.
    If augment=True, training batches are symmetry‐augmented.
    """
    SPLIT_RATIO = 0.8
    MAT_NAME = 'unit_cell_data.mat'

    # 1) load .mat
    data = sio.loadmat(MAT_NAME)['unit_cell_data']
    all_data = np.stack(data[0], axis=0).astype('float32')              # N×50×50
    all_label = np.stack([m.T[num] for m in data[1]], axis=0).astype('float32')  # N×31

    # 2) optional sub‐sampling
    if subset_size is not None:
        all_data = all_data[:subset_size]
        all_label = all_label[:subset_size]

    # 3) train/test split
    split_idx = int(len(all_data) * SPLIT_RATIO)
    Xtr = torch.from_numpy(all_data[:split_idx])
    Ytr = torch.from_numpy(all_label[:split_idx])
    Xte = torch.from_numpy(all_data[split_idx:])
    Yte = torch.from_numpy(all_label[split_idx:])

    # 4) setup augment and collate functions
    aug = CoreSymmetryAugment(max_shift=5)

    def collate_train(batch):
        imgs, labs = zip(*batch)
        imgs = torch.stack(imgs).unsqueeze(1)  # B×1×50×50
        if augment:
            imgs = torch.stack([aug(img) for img in imgs])
        return imgs, torch.stack(labs)

    def collate_test(batch):
        imgs, labs = zip(*batch)
        imgs = torch.stack(imgs).unsqueeze(1)
        return imgs, torch.stack(labs)

    # 5) build DataLoaders
    train_loader = Data.DataLoader(
        Data.TensorDataset(Xtr, Ytr),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_train
    )
    test_loader = Data.DataLoader(
        Data.TensorDataset(Xte, Yte),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_test
    )

    return test_loader, train_loader
