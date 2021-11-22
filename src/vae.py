import torch
from torch import nn, optim
from torch.nn import functional as F

class MLPEncoder(nn.Module):
    def __init__(self,
                 kwargs
                 ):
        super(MLPEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['input_dim'], 32), nn.Tanh(),
            nn.Linear(32, 64), nn.Tanh(),
            nn.Linear(64, kwargs['output_dim'])
        )

    def forward(self, state):
        feature = self.main(state)
        return feature


class MLPDecoder(nn.Module):
    def __init__(self,
                 kwargs
                 ):
        super(MLPDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['input_dim'], 32), nn.Tanh(),
            nn.Linear(32, 64), nn.Tanh(),
            nn.Linear(64, kwargs['output_dim'])
        )

    def forward(self, z):
        state = self.main(z)
        return state

class CNNEncoder(nn.Module):
    def __init__(self,
            kwargs
                 ):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=kwargs['in_channels'], out_channels=32, stride=(2, 2), kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, stride=(2, 2), kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32 * 2, stride=(2, 2), kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(32 * 2)
        self.conv4 = nn.Conv2d(in_channels=32 * 2, out_channels=32 * 2, stride=(2, 2), kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(32 * 2)

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

    def forward(self, state):
        '''
        state: (B, 4, 84, 84)
        z: (B, 64, 4, 4)
        '''
        x = self.lrelu(self.bn1(self.conv1(state)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x.view(x.size()[0], -1)

class CNNDecoder(nn.Module):
    def __init__(self,
                 kwargs
                 ):
        super(CNNDecoder, self).__init__()

        self.linear1 = nn.Linear(kwargs['input_dim'], 64)
        self.linear2 = nn.Linear(64, 1024)

        self.conv5 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=32 * 2, kernel_size=(3, 3), stride=(2, 2))
        self.bn5 = nn.BatchNorm2d(32 * 2)
        self.conv6 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=32 * 2, kernel_size=(3, 3), stride=(2, 2))
        self.bn6 = nn.BatchNorm2d(32 * 2)
        self.conv7 = nn.ConvTranspose2d(in_channels=32 * 2, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(8, 8), stride=(2, 2))
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=kwargs['out_channels'], kernel_size=(1, 1), stride=(1, 1))

        # Leaky relu activation
        self.lrelu = nn.LeakyReLU()

        # Initialize weights using xavier initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.conv9.weight)

    def forward(self, z):
        x = self.lrelu(self.linear1(z))
        x = self.lrelu(self.linear2(x))
        x = x.view((x.size()[0], 64, 4, 4))

        x = self.lrelu(self.bn5(self.conv5(x)))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = self.lrelu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        fake_state = self.conv9(x)

        return fake_state

class VAEBackbone(nn.Module):
    def __init__(self,
                 device,
                 encoder,
                 decoder,
                 hidden_dim,
                 latent_dim
                 ):
        super(VAEBackbone, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        # mean and logvariance of latent variable
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.device = device

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn_like(std, requires_grad=True).to(self.device)

            return epsilon.mul(std).add_(mu)
        else:
            return mu

    def forward(self, obs, training=True):
        z = self.encoder(obs)
        mu = self.mu(z)
        logvar = self.logvar(z)

        z = self.reparameterize(mu, logvar, training=training)

        recon_obs = self.decoder(z)

        return z, mu, logvar, recon_obs

class VAE:
    def __init__(self,
                 device,
                 ob_shape,
                 latent_dim,
                 lambda_kld=1.0,
                 lambda_recon=1.0
                 ):
        self.device = device
        if len(ob_shape) == 3:
            encoder = CNNEncoder(kwargs={'in_channels': ob_shape[0]})
            decoder = CNNDecoder(kwargs={'input_dim': latent_dim, 'out_channels': ob_shape[0]})
            self.vae_backbone = VAEBackbone(
                device=device,
                encoder=encoder,
                decoder=decoder,
                latent_dim=latent_dim,
                hidden_dim=1024
            ).to(self.device)
        else:
            encoder = MLPEncoder(kwargs={'input_dim': ob_shape[0], 'output_dim': 256})
            decoder = MLPDecoder(kwargs={'input_dim': latent_dim, 'output_dim': ob_shape[0]})
            self.vae_backbone = VAEBackbone(
                device=device,
                encoder=encoder,
                decoder=decoder,
                latent_dim=latent_dim,
                hidden_dim=256
            ).to(self.device)

        self.optimizer = optim.Adam(self.vae_backbone.parameters(), lr=5e-4)

        self.lambda_kld = lambda_kld
        self.lambda_recon = lambda_recon

    def get_vae_loss(self, x, fake_x, mu, logvar):
        kld_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = F.mse_loss(x, fake_x)

        return kld_loss, recon_loss, self.lambda_kld * kld_loss + recon_loss

    def train_on_batch(self, obs, training=True):
        self.optimizer.zero_grad()

        obs = obs.to(self.device)
        z, mu, logvar, fake_obs = self.vae_backbone(obs, training)
        kld_loss, recon_loss, vae_loss = self.get_vae_loss(obs, fake_obs, mu, logvar)
        vae_loss.backward()
        self.optimizer.step()

        return kld_loss.item(), recon_loss.item(), vae_loss.item()