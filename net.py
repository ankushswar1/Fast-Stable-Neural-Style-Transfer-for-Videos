import torch
import torch.nn as nn
import random

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

# essentially a reflection of the encoder, upsampling with nearest neighbors
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# pretrained VGG-19 taken from Simonyan paper
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()

        # populate layers of encoder
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        # initialize decoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    # get activation function outputs for loss comp later
    def encode_with_intermediate(self, img):
        results = [img]
        for i in range(4):
            func = getattr(self, 'enc_' + str(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    # final activation
    def encode(self, img):
        for i in range(4):
            img = getattr(self, 'enc_' + str(i+1))(img)
        return img

    def calc_content_loss(self, img, content):
        return self.mse_loss(img, content)

    def calc_style_loss(self, img, style):
        # Use IN to calc style loss
        img_mean, img_std = calc_mean_std(img)
        style_mean, style_std = calc_mean_std(style)
        return self.mse_loss(img_mean, style_mean) + self.mse_loss(img_std, style_std)

    def calc_noise_loss(self, img, style):
        return self.mse_loss(img, style);

    def generate_noisy_input(self, img, noise_range, noise_count):
        image_dims = input.size();
        num_img = image_dims[0];

        # initialize noisy image
        noiseimg = torch.zeros_like(input);

        # approx 1000 pixels for MPI-Sintel input
        noise_count = 0.003 * image_dims[2] * image_dims[3];

        for ct in range(int(noise_count)):
            for img in range(num_img):
                # choose which pixels to add to
                x_idx = random.randrange(image_dims[3])
                y_idx = random.randrange(image_dims[2])

                for ch in range(2):
                    # add noise across all three channels
                    noiseimg[img][ch][y_idx][x_idx] += random.randrange(-noise_range, noise_range);

        # noiseimg gives you a mask, add to input
        return noiseimg + input;


    def forward(self, content, style, alpha=1.0):
        # output pastiche for original input
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat
        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        # output pastiche for noisy input
        content_noise = self.generate_noisy_input(content, 40, 200);
        with torch.no_grad():
            content_noise_feat = self.encode(content_noise)
            t_noise = adain(content_noise_feat, style_feats[-1])
            t_noise = alpha * t_noise + (1 - alpha) * content_noise_feat
            g_t_noise = self.decoder(t_noise)
            g_t_noise_feats = self.encode_with_intermediate(g_t_noise)

        # calculate losses
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_n = self.calc_noise_loss(g_t_feats[-1], g_t_noise_feats[-1])
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])

        # calculate for all filters
        for i in range(1, 4, 1):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s, loss_n
