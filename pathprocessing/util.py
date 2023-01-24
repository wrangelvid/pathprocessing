import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from PIL.Image import Image as pillow_image
from typing import Union

##### STYLE TRANSFER ####


class _ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super(_ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def _gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class _StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(_StyleLoss, self).__init__()
        self.target = _gram_matrix(target_feature).detach()

    def forward(self, input):
        G = _gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class _Normalization(nn.Module):
    def __init__(self, mean, std):
        super(_Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().requires_grad_(True).view(-1, 1, 1)
        self.std = std.clone().detach().requires_grad_(True).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def transfer_style(
    content_img: Union[pillow_image, str],
    style_img: Union[pillow_image, str],
    im_size: int = 256,
    num_steps: int = 300,
    style_weight: int = 1000000,
    content_weight: int = 1,
    verbose: bool = True,
) -> pillow_image:
    """Run neural style transfer.

    The transfer loss is the sum of the content and style loss.

    Args:
        content_img: Either str to path or Image object that we wish to stylize.
        style_img: Either str to path or Image object that determines the style.
        im_size: Scales the content image to this size.
        num_steps: Number of optimization steps.
        style_weight: Scales the style score for the loss.
        content_weight: Scales the content score for the loss.
        verbose: Prints out progress info.

    Returns:
        A stylized image.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load images in.
    if device == "cpu" and im_size > 256:
        raise Exception("Need a GPU to process images larger than 256 pixels.")

    loader = transforms.Compose(
        [transforms.Resize(im_size), transforms.ToTensor()]  # scale imported image
    )  # transform it into a torch tensor

    if type(content_img) is str:
        content_img = Image.open(content_img).convert('RGB')
    content_img = loader(content_img).unsqueeze(0).to(device, torch.float)

    loader = transforms.Compose(
        [transforms.Resize(content_img.size()[-2:]), transforms.ToTensor()]  # scale imported image
    )  # transform it into a torch tensor

    if type(style_img) is str:
        style_img = Image.open(style_img).convert('RGB')
    style_img = loader(style_img).unsqueeze(0).to(device, torch.float)

    assert style_img.size() == content_img.size()

    input_img = content_img.clone()  # could also start with white noise.

    if verbose:
        print("Building the style transfer model..")
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # normalization module
    normalization = _Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = _ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = _StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], _ContentLoss) or isinstance(model[i], _StyleLoss):
            break

    model = model[: (i + 1)]

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    # get input optimizer.
    optimizer = optim.LBFGS([input_img])

    if verbose:
        print("Optimizing..")
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if verbose and run[0] % 50 == 0:
                print(f"run {run[0]}:")
                print("Style Loss : {:4f} Content Loss: {:4f}".format(style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    unloader = transforms.ToPILImage()  # reconvert into PIL image
    input_img = input_img.cpu().clone()  # we clone the tensor to not do changes on it
    input_img = input_img.squeeze(0)  # remove the fake batch dimension
    return unloader(input_img)


##### Informative Drawing #####
class _ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(_ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class _Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(_Generator, self).__init__()

        # Initial convolution block
        model0 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [_ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


def image_to_drawing(
    img: Union[pillow_image, str],
    im_size: int = 256,
    style: str = "contour",
    input_nc: int = 3,
    output_nc: int = 1,
    n_blocks: int = 3,
) -> pillow_image:
    """Creates a line drawing from an image.

    Implements
    "Learning to generate line drawings that convey geometry and semantics"
    by Caroline Chan, Fredo Duran, Phillip Isola.

    Does not predict depth or reconstruct image.

    Args:
        img: The image to extract the contours from.
        im_size: Size of the image (assumes its a square).
        style: The desired style must be either contour, anime or sketch.
        input_nc: Number of channels of input data.
        output_nc: Number of channels of output data.
        n_blocks: Number of resnet blocks for generator.
    
    Returns:
        A line drawing image in grayscale.
    """
    allowed_style = ["contour", "anime", "sketch"]
    if style not in allowed_style:
        raise Exception(f"Provided style: {style}. But must be either {allowed_style}")

    if type(img) is str:
        img = Image.open(img).convert('RGB')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("./checkpoints", f"netG_{style}.pth")

    with torch.no_grad():
        # Load model.
        net_G = _Generator(input_nc, output_nc, n_blocks).to(device)
        net_G.load_state_dict(torch.load(model_path, map_location=device))
        net_G.eval()

        # Load image.
        loader = transforms.Compose(
            [transforms.Resize(im_size, transforms.InterpolationMode.BICUBIC), transforms.ToTensor()]
        )

        img = Variable(loader(img)).to(device)

        # Predict.
        output_img = net_G(img)
        output_img.clamp_(0, 1)

        unloader = transforms.ToPILImage()  # reconvert into PIL image
        output_img = output_img.cpu().clone()  # we clone the tensor to not do changes on it
        output_img = output_img.squeeze(0)  # remove the fake batch dimension
        return unloader(output_img)
