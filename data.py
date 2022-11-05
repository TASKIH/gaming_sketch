# Mostly copied from Anime2Sketch.
# https://github.com/Mukosame/Anime2Sketch

from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

try:
    from torchvision.transforms import InterpolationMode
    bic = InterpolationMode.BICUBIC
except ImportError:
    bic = Image.BICUBIC

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def get_transform(load_size=0):
    transform_list = []
    if load_size > 0:
        output_size = [load_size, load_size]
        transform_list.append(transforms.Resize(output_size, bic))

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def read_img_path(path, load_size):
    """read tensors from a given image path
    Parameters:
        path (str)     -- input image path
        load_size(int) -- the input size. If <= 0, don't resize
    """
    img = Image.open(path).convert('RGB')
    aus_resize = None
    if load_size > 0:
        aus_resize = img.size
    transform = get_transform(load_size=load_size)
    image = transform(img)
    return image.unsqueeze(0), aus_resize


def tensor_to_img(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)