from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10

# set size = 32 for CIFAR10, size = 272 for ADP
size = 272

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(), ])

test_transform = transforms.Compose([
    transforms.Resize(size=size),
    transforms.ToTensor(), ])


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target
