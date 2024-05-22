import torch
import torch.nn as nn
import torch.nn.functional as F

class MediumCNN(nn.Module):
    def __init__(self, input_shape):
        super(MediumCNN, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding="same")
        #self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        #self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        #self.conv4 = nn.Conv2d(128, 256, 3, padding="same")
        #self.conv5 = nn.Conv2d(256, 512, 3, padding="same")
        #self.conv6 = nn.Conv2d(512, 1024, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.fc = nn.Linear(1024, 1024)

        self.tconv1 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.tconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.tconv4 = nn.ConvTranspose2d(128, 64, 3, stride=4, padding=1, output_padding=3)
        self.tconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.tconv6 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.tconv7 = nn.ConvTranspose2d(16, 8, 3, padding="same")
        self.tconv8 = nn.ConvTranspose2d(8, 4, 3, padding="same")
        self.tconv9 = nn.ConvTranspose2d(4, 2, 3, padding="same")
        self.tconv10 = nn.ConvTranspose2d(2, 1, 3, padding="same")

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.maxpool(x)

        x = F.elu(self.conv3(x))
        x = self.maxpool(x)

        x = F.elu(self.conv4(x))
        x = self.maxpool(x)

        x = F.elu(self.conv5(x))
        x = self.maxpool(x)

        #x = F.elu(self.conv6(x))
        #x = self.maxpool4(x)

        x = torch.flatten(x, 1)
        x = F.elu(self.fc(x))

        x = x.view(-1, 1024, 1, 1)  # Reshape for the transposed convolutions
        x = F.elu(self.tconv1(x))
        x = F.elu(self.tconv2(x))
        x = F.elu(self.tconv3(x))
        x = F.elu(self.tconv4(x))
        x = F.elu(self.tconv5(x))
        x = F.elu(self.tconv6(x))
        x = F.elu(self.tconv7(x))
        x = F.elu(self.tconv8(x))
        x = F.elu(self.tconv9(x))
        x = self.tconv10(x)  # Linear activation for the output layer

        return x

def build_model(input_shape):
    return MediumCNN(input_shape)

