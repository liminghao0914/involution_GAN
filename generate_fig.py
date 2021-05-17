import torch
from utils1 import *
from torchvision.utils import save_image
import torch.nn as nn
from sagan_models import Generator_SA, Discriminator_SA
from igan_models import Generator_INV, Discriminator_INV
from dcgan_models import Generator_DC, Discriminator_DC
from gan_models import Generator_MLP, Discriminator_MLP

path = 'models/mnist_sagan_5.5/'
listdir = os.listdir(path)
for i, pth in enumerate(listdir):
    if "_G" in pth:
        epoch = pth.split("_G")[0]
        print(i)
        model = torch.load(os.path.join(path, pth))
        # print(model)
        torch.manual_seed(0)
        # print(torch.randn(1,128))
        if not os.path.exists(os.path.join(path, 'img/' + epoch)):
            os.makedirs(os.path.join(path, 'img/' + epoch))
        for i in range(1000):
            fixed_z = tensor2var(torch.randn(1, 128))
            fake_images, _, _ = model(fixed_z)
            fake_images_new = fake_images[0].view(1, 3, 64, 64)
            save_image(denorm(fake_images_new.data), os.path.join(path, 'img/' + epoch + '/' + str(i) + '.jpg'))
