
import os
import time
import torch
import datetime
import json
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from utils1 import *

from sagan_models import Generator_SA, Discriminator_SA
from igan_models_case1 import Generator_INV1, Discriminator_INV1
from igan_models_case2 import Generator_INV2, Discriminator_INV2
from igan_models_case3 import Generator_INV3, Discriminator_INV3
from igan_models_case4 import Generator_INV4, Discriminator_INV4
from dcgan_models import Generator_DC, Discriminator_DC
from gan_models import Generator_MLP, Discriminator_MLP
from utils import *




class Trainer(object):
    def __init__(self, data_loader, config):
        # Data record
        self.record = {
            'lossd': [],
            'lossg': [],
            'gp': [],
        }

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss
        self.case = config.case
        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.rgb_channel = 3
        if self.dataset == 'mnist':
            self.rgb_channel = 1

        self.build_model()

        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []


        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()


    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            lossd = []
            lossg = []
            gp = []
            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            d_out_real,dr1,dr2 = self.D(real_images)


            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))

            fake_images,gf1,gf2 = self.G(z)
            d_out_fake,df1,df2 = self.D(fake_images)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            lossd = d_loss.item()
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_,_ = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
                gp = d_loss_gp.item()
                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            #if (step + 1) % 5 == start:

            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,_,_ = self.G(z)

            # Compute loss with fake images
            g_out_fake,_,_ = self.D(fake_images)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()
            lossg = g_loss_fake.item()
            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()


            # Print out log info
            if (step + 1) % self.log_step == 0:

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                if self.model == 'sagan':
                    print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], loss_G: {:.4f},loss_D: {:.4f},penalty: {:.4f}, "
                          " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                          format(elapsed, step + 1, self.total_step, (step + 1),
                                 self.total_step, lossg, lossd, gp,
                                 self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item()))
                if self.model in ['dcgan', 'gan', 'igan']:
                    print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], loss_G: {:.4f}, loss_D: {:.4f}, penalty: {:.4f}".
                          format(elapsed, step + 1, self.total_step, (step + 1),
                                 self.total_step, lossg, lossd, gp))
                self.record["lossd"].append(lossd)
                self.record["lossg"].append(lossg)
                self.record["gp"].append(gp)

                with open(os.path.join(self.log_path, "loss.log"), 'w') as f:
                    json.dump(self.record, f)

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_,_= self.G(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.jpg'.format(step + 1)))

            if (step+1) % model_save_step==0:
                torch.save(self.G,
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D,
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):
        print(self.case)

        if self.model == 'sagan':
            self.G = Generator_SA(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim, self.rgb_channel).cuda()
            self.D = Discriminator_SA(self.batch_size,self.imsize, self.d_conv_dim, self.rgb_channel).cuda()
        elif self.model == 'dcgan':
            self.G = Generator_DC(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim, self.rgb_channel).cuda()
            self.D = Discriminator_DC(self.batch_size,self.imsize, self.d_conv_dim, self.rgb_channel).cuda()
        elif self.model == 'gan':
            self.G = Generator_MLP(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim, self.rgb_channel).cuda()
            self.D = Discriminator_MLP(self.batch_size,self.imsize, self.d_conv_dim, self.rgb_channel).cuda()
        elif self.model == 'igan':

            if self.case == 1:
                self.G = Generator_INV1(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim, self.rgb_channel).cuda()
                self.D = Discriminator_INV1(self.batch_size,self.imsize, self.d_conv_dim, self.rgb_channel).cuda()
            elif self.case == 2:
                self.G = Generator_INV2(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim, self.rgb_channel).cuda()
                self.D = Discriminator_INV2(self.batch_size,self.imsize, self.d_conv_dim, self.rgb_channel).cuda()
            elif self.case == 3:
                self.G = Generator_INV3(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim, self.rgb_channel).cuda()
                self.D = Discriminator_INV3(self.batch_size, self.imsize, self.d_conv_dim, self.rgb_channel).cuda()
            elif self.case == 4:
                self.G = Generator_INV4(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim,
                                        self.rgb_channel).cuda()
                self.D = Discriminator_INV4(self.batch_size, self.imsize, self.d_conv_dim, self.rgb_channel).cuda()

        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))