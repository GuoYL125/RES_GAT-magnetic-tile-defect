import os
import pickle
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset


class Mydataset(VisionDataset):
    """Custom dataset for magnetic tile defect classification.

    Expects preprocessed batch files in:
        {root}/batch_save_train_masks/data_batch_MT_*_masks

    Each batch file contains dicts with 'data' (images) and 'labels'.
    """
    base_folder = 'batch_save_train_masks'

    train_list = [
        ['data_batch_MT_Blowhole_masks', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_MT_Crack_masks', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_MT_Break_masks', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ['data_batch_MT_Fray_masks', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_MT_Uneven_masks', '634d18415352ddfa80567beed471001a'],
    ]

    test_list = [
        ['data_batch_airbus', 'c99cafc152244af753f735de768cd75f'],
    ]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(Mydataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        downloaded_list = self.train_list if self.train else self.test_list

        self.data: Any = []
        self.targets = []

        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 224, 224)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
