# -*- coding: utf-8 -*-

"""
Created on January 28, 2021
@author: Siqi Miao
"""

import os
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST


class VAE(nn.Module):

    def __init__(self, in_features, latent_size, y_size=0):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        self.encoder_forward = nn.Sequential(
            nn.Linear(in_features + y_size, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, self.latent_size * 2)  #注意乘以2
        )

        self.decoder_forward = nn.Sequential(
            nn.Linear(self.latent_size + y_size, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )

    def encoder(self, X):
        out = self.encoder_forward(X)
        mu = out[:, :self.latent_size]
        log_var = out[:, self.latent_size:]
        return mu, log_var

    def decoder(self, z):
        mu_prime = self.decoder_forward(z)
        return mu_prime

    def reparameterization(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon * torch.sqrt(log_var.exp())
        return z

    def loss(self, X, mu_prime, mu, log_var):
        # reconstruction_loss = F.mse_loss(mu_prime, X, reduction='mean') is wrong!
        reconstruction_loss = torch.mean(torch.square(X - mu_prime).sum(dim=1))

        latent_loss = torch.mean(0.5 * (log_var.exp() + torch.square(mu) - log_var).sum(dim=1))   #注意，第一个输出是log var方
        return reconstruction_loss + latent_loss

    def forward(self, X, *args, **kwargs):
        mu, log_var = self.encoder(X)
        z = self.reparameterization(mu, log_var)
        mu_prime = self.decoder(z)
        return mu_prime, mu, log_var

class CVAE(VAE):

    def __init__(self, in_features, latent_size, y_size):
        super(CVAE, self).__init__(in_features, latent_size, y_size)

    def forward(self, X, y=None, *args, **kwargs):
        y = y.to(next(self.parameters()).device)  #使用和parameters相同的设备
        X_given_Y = torch.cat((X, y.unsqueeze(1)), dim=1) #按维数1（行）拼接向量 横着   unsqueeze 扩展为行维度  列数变行数

        mu, log_var = self.encoder(X_given_Y)
        z = self.reparameterization(mu, log_var)
        z_given_Y = torch.cat((z, y.unsqueeze(1)), dim=1)

        mu_prime_given_Y = self.decoder(z_given_Y)
        return mu_prime_given_Y, mu, log_var


def train(model, optimizer, data_loader, device, name='VAE'):
    model.train()

    total_loss = 0
    pbar = tqdm(data_loader)
    for X, y in pbar:
        batch_size = X.shape[0]
        X = X.view(batch_size, -1).to(device) #调整维度为 batchsize:?   三维拉为二维度  图片拉为一维向量
        model.zero_grad()

        if name == 'VAE':
            mu_prime, mu, log_var = model(X)
        else:
            mu_prime, mu, log_var = model(X, y)

        loss = model.loss(X.view(batch_size, -1), mu_prime, mu, log_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))

    return total_loss / len(data_loader)


@torch.no_grad()
def save_res(vae, cvae, data, latent_size, device):
    num_classes = len(data.classes)

    # raw samples from dataset
    out = []
    for i in range(num_classes):
        img = data.data[torch.where(data.targets == i)[0][:num_classes]]  #提取data中标签为 的数据
        out.append(img)
    out = torch.stack(out).transpose(0, 1).reshape(-1, 1, 28, 28)
    save_image(out.float(), './img/raw_samples.png', nrow=num_classes, normalize=True)

    # samples generated by vanilla VAE
    z = torch.randn(num_classes ** 2, latent_size).to(device)   # u和lou  两组
    out = vae.decoder(z)
    save_image(out.view(-1, 1, 28, 28), './img/vae_samples.png', nrow=num_classes)

    # sample generated by CVAE
    z = torch.randn(num_classes ** 2, latent_size).to(device) #也是正态
    y = torch.arange(num_classes).repeat(num_classes).to(device)  # 按1步长生成张量  张量复制
    z_given_Y = torch.cat((z, y.unsqueeze(1)), dim=1)
    out = cvae.decoder(z_given_Y)
    save_image(out.view(-1, 1, 28, 28), './img/cvae_samples.png', nrow=num_classes)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    batch_size = 256 * 4
    epochs = 50
    latent_size = 64
    in_features = 28 * 28
    lr = 0.001

    data = MNIST('../../dataset/', download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # train VAE
    vae = VAE(in_features, latent_size).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)

    print('Start Training VAE...')
    for epoch in range(1, 1 + epochs):
        loss = train(vae, optimizer, data_loader, device, name='VAE')
        print("Epochs: {epoch}, AvgLoss: {loss:.4f}".format(epoch=epoch, loss=loss))
    print('Training for VAE has been done.')

    # train VCAE
    cvae = CVAE(in_features, latent_size, y_size=1).to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=lr)

    print('Start Training CVAE...')
    for epoch in range(1, 1 + epochs):
        loss = train(cvae, optimizer, data_loader, device, name='CVAE')
        print("Epochs: {epoch}, AvgLoss: {loss:.4f}".format(epoch=epoch, loss=loss))
    print('Training for CVAE has been done.')

    save_res(vae, cvae, data, latent_size, device)


if __name__ == '__main__':
    main()