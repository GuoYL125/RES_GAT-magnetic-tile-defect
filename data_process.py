import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def Myloader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def data_gyl(label, data_path):
    """Parse label file: each line is 'filename label'."""
    data = []
    with open(label, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            if len(parts) == 2:
                data.append([data_path + parts[0].strip(), int(parts[1].strip())])
    return data


def load_data():
    """Load CNN training/validation/test data from dataset-cnn folder."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    data1_path = './dataset/dataset-cnn/MT_Blowhole/'
    data2_path = './dataset/dataset-cnn/MT_Break/'
    data3_path = './dataset/dataset-cnn/MT_Crack/'

    label1 = './dataset/dataset-cnn/image_train_MT_Blowhole_list.txt'
    label2 = './dataset/dataset-cnn/image_train_MT_Break_list.txt'
    label3 = './dataset/dataset-cnn/image_train_MT_Crack_list.txt'

    data = data_gyl(label1, data1_path) + data_gyl(label2, data2_path) + data_gyl(label3, data3_path)
    np.random.shuffle(data)

    num_train = int(len(data) * 0.7)
    num_val = int(len(data) * 0.1)
    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]

    Dtr = DataLoader(MyDataset(train_data, transform, Myloader), batch_size=5, shuffle=False, num_workers=1)
    Val = DataLoader(MyDataset(val_data, transform, Myloader), batch_size=5, shuffle=False, num_workers=1)
    Dte = DataLoader(MyDataset(test_data, transform, Myloader), batch_size=5, shuffle=False, num_workers=1)

    return Dtr, Val, Dte
