import torch
import torchvision.datasets as dsets
from torchvision import transforms, datasets
from PIL import Image
import os

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, centercrop,  resize, totensor, normalize):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(64))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform


    def load_celeb(self):
        transforms = self.transform(False, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/resized_celebA/celebA/', transform=transforms)
        return dataset

    def load_getbaby(self):
        transforms = self.transform(False, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/getbaby/mt/', transform=transforms)
        return dataset

    def load_anime(self):
        transforms = self.transform(False, True, True, True)
        dataset = dsets.ImageFolder(self.path + '/resized_Anime', transform=transforms)
        return dataset

    def load_mnist(self):
        transform = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'anime':
            dataset = self.load_anime()
        elif self.dataset == 'getbaby':
            dataset = self.load_getbaby()
        elif self.dataset == 'mnist':
            dataset = self.load_mnist()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader
