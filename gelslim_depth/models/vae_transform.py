import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List


# write a simple convolutional neural network to classify images
class VAETransformation(nn.Module):
    def __init__(self, input_dimensions: Tuple[int,int], num_input_channels: int, num_output_channels: int, num_CNN_layers: int, num_FCpre_layers: int, CNN_dimensions: List, FCpre_dimensions: List, kernel_size: int, maxpool_size: int, latent_dimension: int, activation_func: str = "relu", device=None):
        super().__init__()
        if num_CNN_layers < 1:
            raise ValueError("num_CNN_layers must be at least 1")
        if num_FCpre_layers < 0:
            raise ValueError("num_FCpre_layers must be at least 0")
        if len(CNN_dimensions) != num_CNN_layers:
            raise ValueError("CNN_dimensions must have length num_CNN_layers")
        if len(FCpre_dimensions) != num_FCpre_layers:
            raise ValueError("FCpre_dimensions must have length num_FCpre_layers")
        #

        if activation_func == "relu":
            self.activation_func = nn.ReLU()
        elif activation_func == "tanh":
            self.activation_func = nn.Tanh()
        elif activation_func == "mish":
            self.activation_func = nn.Mish()
        else:
            raise ValueError("activation_function must be 'relu' or 'tanh' or 'mish'")
        
        self.maxpool = nn.MaxPool2d(maxpool_size)

        print("input_dimensions: "+str(input_dimensions))

        #convolutional layers
        conv_layers = []
        in_channels = num_input_channels
        for i in range(num_CNN_layers):
            out_channels = CNN_dimensions[i]
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        self.conv_layers = nn.ModuleList(conv_layers)

        #calculate the size of the flattened output from the convolutional layers
        test_input = torch.zeros(1, num_input_channels, input_dimensions[0], input_dimensions[1])
        test_output = self.conv_layers[0](test_input)
        test_output = self.activation_func(test_output)
        test_output = self.maxpool(test_output)
        for i in range(1, num_CNN_layers):
            print(str(i)+": "+str(self.conv_layers[i]))
            test_output = self.conv_layers[i](test_output)
            test_output = self.activation_func(test_output)
            test_output = self.maxpool(test_output)
        output_conv_size = test_output.size()
        test_output = torch.flatten(test_output, 1)
        flattened_size = test_output.size()[1]
        in_features = flattened_size

        fc_pre_layers = []
        for i in range(num_FCpre_layers):
            out_features = FCpre_dimensions[i]
            fc_pre_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.fc_pre_layers = nn.ModuleList(fc_pre_layers)

        self.fc_mu = nn.Linear(in_features, latent_dimension)

        self.fc_sigma = nn.Linear(in_features, latent_dimension)

        #reverse_fc_layers
        reverse_FC_layers = []
        in_features = latent_dimension
        for i in range(num_FCpre_layers):
            out_features = FCpre_dimensions[num_FCpre_layers-1-i]
            reverse_FC_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
        self.reverse_fc_pre_layers = nn.ModuleList(reverse_FC_layers)

        self.back_to_flattened_size = nn.Linear(in_features, flattened_size)

        #self.unflatten is a function that takes in a 1D tensor of size flattened_size and returns a 2D tensor of size output_conv_size, using view
        self.unflatten = lambda x,batch_size: x.view(torch.Size([batch_size])+output_conv_size[1:])

        #up conv
        upconv_layers = []
        in_channels = CNN_dimensions[-1]
        for i in range(1,num_CNN_layers):
            out_channels = CNN_dimensions[num_CNN_layers-1-i]
            upconv_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size+1, maxpool_size))
            in_channels = out_channels
        self.upconv_layers = nn.ModuleList(upconv_layers)

        self.final_upconv = nn.ConvTranspose2d(in_channels, num_output_channels, kernel_size+1, maxpool_size)

        #fix size
        self.fix_size = lambda x: F.interpolate(x, size=(input_dimensions[0], input_dimensions[1]), mode='area')

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        #input = x.clone()
        original_size = x.size()
        input = x.clone()
        #import pdb; pdb.set_trace()
        for conv_layer in self.conv_layers:
            x = self.activation_func(conv_layer(x))
            x = self.maxpool(x)
        x = torch.flatten(x, 1)
        #flatten_output = x.clone()
        
        for fc_layer in self.fc_pre_layers:
            x = self.activation_func(fc_layer(x))
        #import pdb; pdb.set_trace()
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))
        latent = mu + sigma*self.N.sample(mu.shape)

        latent_vector = latent.clone()

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()

        for fc_layer in self.reverse_fc_pre_layers:
            latent = self.activation_func(fc_layer(latent))

        output = self.back_to_flattened_size(latent)
        
        output = self.unflatten(output, original_size[0])
        for upconv_layer in self.upconv_layers:
            output = upconv_layer(output)
            output = self.activation_func(output)
        output = self.final_upconv(output)
        output = self.fix_size(output)

        return output, latent_vector