import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch
import torchvision
import os
from torch import optim
from torch.autograd import Variable
from model import D
from model import G

num_epochs = 20
batch_size = 100
z_dim = 100
x_dim = 64
sample_size = 100
lr = 0.0002
log_step = 10
sample_step = 500
sample_path = './samples'
model_path = './models'

img_size = 64
transform = transforms.Compose([
    transforms.Scale(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = dataset.MNIST(root='./data/',
                              train=True,
                              transform=transform,
                              download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

generator = G()
discriminator = D()

g_optimizer = optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))

d_optimizer = optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def reset_grad():
    """Zero the gradient buffers."""
    discriminator.zero_grad()
    generator.zero_grad()


def denorm(x):
    """Convert range (-1, 1) to (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)


def sample():
    g_path = os.path.join(model_path, 'generator-%d.pkl' % (num_epochs))
    d_path = os.path.join(model_path, 'discriminator-%d.pkl' % (num_epochs))
    generator.load_state_dict(torch.load(g_path))
    discriminator.load_state_dict(torch.load(d_path))
    generator.eval()
    discriminator.eval()

    # Sample the images
    noise = to_variable(torch.randn(sample_size, z_dim))
    fake_images = generator(noise)
    sample = os.path.join(sample_path, 'fake_samples-final.png')
    torchvision.utils.save_image(denorm(fake_images.data), sample, nrow=12)
    print("Saved sampled images to '%s'" % sample)


fixed_noise = to_variable(torch.randn(batch_size, z_dim))
fixed_noise = fixed_noise.view([-1, z_dim, 1, 1])
total_step = len(train_loader)

ones_label = Variable(torch.ones(batch_size))
zeros_label = Variable(torch.zeros(batch_size))

for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_loader):
        x = to_variable(x)
        z = to_variable(torch.randn(batch_size, z_dim))
        z = z.view(-1, z_dim, 1, 1)

        outputs = discriminator.forward(x)
        real_loss = torch.mean((outputs - 1) ** 2)

        fake_img = generator.forward(z)
        outputs = discriminator.forward(fake_img)

        fake_loss = torch.mean(outputs ** 2)

        # how torch do training

        d_loss = real_loss + fake_loss
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        z = to_variable(torch.randn(batch_size, z_dim))
        z = z.view(-1, z_dim, 1, 1)
        fake_img = generator(z)
        outputs = discriminator(fake_img)
        g_loss = torch.mean((outputs - 1) ** 2)

        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % log_step == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_real_loss: %.4f, '
                  'd_fake_loss: %.4f, g_loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, total_step,
                     real_loss.data[0], fake_loss.data[0], g_loss.data[0]))

        # save the sampled images
        if (i + 1) % sample_step == 0:
            fake_images = generator(fixed_noise)
            torchvision.utils.save_image(denorm(fake_images.data),
                                         os.path.join(sample_path,
                                                      'fake_samples-%d-%d.png' % (
                                                          epoch + 1, i + 1)))

    g_path = os.path.join(model_path, 'generator-%d.pkl' % (epoch + 1))
    d_path = os.path.join(model_path, 'discriminator-%d.pkl' % (epoch + 1))
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)
