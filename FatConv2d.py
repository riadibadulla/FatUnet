from torch import nn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import math
import torch.nn.functional as F
class FatConv2d(nn.Module):
    """A custom Optical Layer class
    :param input_channels: Number of input channels of the layer
    :type input_channels: int
    :param output_channels: Number of output channels of the layer
    :type outout_channels: int
    :param kernel_size: size of the squared kernel
    :type kernel_size: int
    :param pseudo_negativity: Should the layer use pseudo negativity (decreases the computation time twice)
    :type pseudo_negativity: bool
    :param input_size: Layer accepts only square inputs, this is the size of the sqaured input
    :type input_size: int
    """

    def __init__(self,input_channels,output_channels,kernel_size,is_bias=True,input_size=28):
        super().__init__()
        self.fatconv = nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=kernel_size,padding="same")
        self.kernel_size = kernel_size
        # self.input_channels, self.output_channels = input_channels, output_channels
        # self.kernel_size = kernel_size
        # kernel = torch.Tensor(output_channels,input_channels,kernel_size,kernel_size)
        # self.kernel = nn.Parameter(kernel)
        # nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        # self.is_bias = is_bias
        # if is_bias:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        #     bound = 1 / math.sqrt(fan_in)
        #     bias = torch.Tensor(output_channels)
        #     self.bias = nn.Parameter(bias)
        #     nn.init.uniform_(self.bias, -bound, bound)
        # # self.input_size =input_size

    def process_inputs(self,img):
        """Takes to tensors of image and kernel and pads image or kernel depending on which one is larger
        :param img: Image tensor in (batch_size, input_channels, x, y) shape
        :type img: torch.Tensor
        :param kernel: kernel in (output_channels, input_channels, x ,y) shape
        :type kernel: torch.Tensor
        :return: image and kernel of the same size after padding one of them
        :rtype: torch.Tensor, torch.Tensor
        """
        # img = torch.nn.functional.pad(img, (2, 2, 2, 2))
        if img[0, 0, 0].shape == self.kernel_size:
            return img
        size_of_image = img.shape[2]
        padding_size = abs(size_of_image - self.kernel_size) // 2
        img = torch.nn.functional.pad(img, (padding_size, padding_size, padding_size, padding_size))
        if img[0, 0, 0].shape != self.kernel_size:
            small = torch.nn.functional.pad(img, (0, 1, 0, 1))
        return img

    def forward(self,input):
        """Forward pass of the Optical convolution. It accepts the input tensor of shape (batch_size, input_channels, x, y)
        and takes the weights stored in this class in (output_channels, input_channels, x ,y) shape. Pad them to make
        both tensors same size. Then it turns both input and weight tensor to the same shape
        (batch_size, output_channels, input_channels, x,y) and performs the optical convolution using both tensors.
        Finally sums the output across the input channels to form the output of
        (batch_size, output_channels, x,y) shape.
        :param input: Image tensor in (batch_size, input_channels, x, y) shape
        :type input: torch.Tensor
        :return: output of (batch_size, output_channels, x, y) shape
        :rtype: torch.Tensor
        """
        # batch_size = input.size(dim=0)
        #Padding either input or kernel
        # print("fat conv input", input.shape)
        # print("fat conv kernel", self.kernel.shape)
        input = self.process_inputs(input)
        # print("fat conv input after padding: ", input.shape)
        # print("fat conv kernel after padding: ", kernel.shape)
        # input = input.repeat(1,self.output_channels,1,1)
        # input = torch.reshape(input,(batch_size,self.output_channels,self.input_channels,self.beam_size_px,self.beam_size_px))
        # kernel = kernel.repeat(batch_size,1,1,1)
        # kernel = torch.reshape(kernel,(batch_size,self.output_channels,self.input_channels,self.beam_size_px,self.beam_size_px))
        # output = self.opt.optConv2d(input, kernel, pseudo_negativity=self.pseudo_negativity)
        # output = torch.sum(output, dim=2,dtype=torch.float32)
        output = self.fatconv(input)
        # print("fat conv output", output.shape)
        #Upadding the input
        # if self.kernel_size>self.input_size:
        #     output=output[:,:,self.padding_size:self.padding_size+self.input_size,self.padding_size:self.padding_size+self.input_size]
        #add bias
        # if self.is_bias:
        #     output += self.bias.repeat(batch_size,1,1,1)
        return output