"""
Created on Wed Apr 29 16:11:20 2020

@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x

class FlexExtractor:
    def __init__(self, model, target_layer):
        """
        Initialize the CAM extractor.
        """
        self.model = model
        self.target_layer = target_layer
        self.conv_output = None

    def forward_pass_on_convolutions(self, x):
        conv_output = None
        
        for name, module in self.model.named_modules():
            x = module(x)  # Process every layer starting from the very first
            print("Processing through:", name)
            
            # Check if the target layer is reached
            if name == self.target_layer:
                conv_output = x  # Capture the output at the target layer
                print("Target layer found and processed:", name)
                break  # Exit loop after processing the target layer

        if conv_output is None:
            print("Warning: Target layer not found. Ensure the target_layer is correct.")
        
        return conv_output, x




    def forward_pass(self, x):
        """
        Performs a forward pass on the model.
        """
        # print("PUMASOK SA FORWARD PASS NG FLEXEXTRACTOR")
        # Assuming the target layer is part of the convolutional base
        # print(x)
        conv_output, x = self.forward_pass_on_convolutions(x)
        # Additional handling for models with separate classifier part,
        # if applicable, similar to the original approach.
        # print(x)
        return conv_output, x

class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        # if resnet or densenet, forward pass is different
        if 'ResNet' in self.model.__class__.__name__:
            self.extractor = FlexExtractor(self.model, target_layer)
        elif 'DenseNet' in self.model.__class__.__name__:
            self.extractor = FlexExtractor(self.model, target_layer)
        else:
            self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        # print(input_image)
        conv_output, model_output = self.extractor.forward_pass(input_image)

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.LANCZOS))/255
        return cam


# if __name__ == '__main__':
#     # Get params
#     target_example = 0  # Snake
#     (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#         get_example_params(target_example)
#     # Score cam
#     score_cam = ScoreCam(pretrained_model, target_layer=11)
#     # Generate cam mask
#     cam = score_cam.generate_cam(prep_img, target_class)
#     # Save mask
#     save_class_activation_images(original_image, cam, file_name_to_export)
#     print('Score cam completed')
