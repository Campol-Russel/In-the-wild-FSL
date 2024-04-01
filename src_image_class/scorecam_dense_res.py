import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image

class ScoreCAM:
    """
    Implements Score-CAM, a class activation mapping method that uses the
    model's feature maps and scores to generate a class-specific activation map.
    
    Attributes:
        model (torch.nn.Module): The neural network model.
        target_layer (torch.nn.modules.conv.Conv2d): The target convolutional layer from which
            feature maps are extracted.
        feature_maps (torch.Tensor): Stores the feature maps from the target layer.
        model_output (torch.Tensor): Stores the output of the model.
        hook_handles (list): Stores handles to the registered hooks, allowing for their removal.
    
    """

    def __init__(self, model, target_layer):
        """
        Initializes the ScoreCAM instance.
        
        Args:
            model (torch.nn.Module): The model to be analyzed.
            target_layer (torch.nn.modules.conv.Conv2d): The convolutional layer whose feature maps
                are to be used for generating the CAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.feature_maps = None
        self.model_output = None

        # Register hook for the target layer
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_feature_maps))
        self.model.register_forward_hook(self.save_output)

    def save_feature_maps(self, module, input, output):
        """
        Hook to save the feature maps from the target layer.
        
        Args:
            module: The module being hooked.
            input: The input to the module.
            output: The output from the module (feature maps).
        """
        self.feature_maps = output.detach()

    def save_output(self, module, input, output):
        """
        Hook to save the model's output.
        
        Args:
            module: The module being hooked.
            input: The input to the module.
            output: The output from the module (model output).
        """
        self.model_output = output.detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generates the Class Activation Map (CAM) for a specific class.
        
        Args:
            input_image (torch.Tensor): The input image tensor.
            target_class (int, optional): The target class for which the CAM is generated.
                If None, the class with the highest score in the model's output is used.
                
        Returns:
            numpy.ndarray: The generated CAM as a NumPy array.
        """
        self.model(input_image)
        
        if target_class is None:
            target_class = self.model_output.argmax(dim=1).item()

        # Initialize the CAM to zeros
        cam = torch.zeros(input_image.shape[2:], device=input_image.device)
        
        # Process each feature map
        for i in range(self.feature_maps.shape[1]):
            fmap = self.feature_maps[0, i, :, :]
            if torch.max(fmap) == torch.min(fmap):
                continue

            # Normalize the feature map
            fmap = (fmap - torch.min(fmap)) / (torch.max(fmap) - torch.min(fmap))
            fmap_upscaled = F.interpolate(fmap.unsqueeze(0).unsqueeze(0), 
                                           size=input_image.shape[2:], 
                                           mode='bilinear', 
                                           align_corners=False).squeeze()

            # Apply the feature map as a mask
            masked_input = input_image.clone()
            for j in range(masked_input.shape[1]):
                masked_input[:, j, :, :] *= fmap_upscaled

            output = self.model(masked_input)
            score = F.softmax(output, dim=1)[0, target_class]
            cam += score.item() * fmap_upscaled.cpu()

        cam = cam / torch.max(cam)  # Normalize the CAM

        return cam.cpu().numpy()

    def clear_hooks(self):
        """
        Removes the hooks from the model.
        """
        for handle in self.hook_handles:
            handle.remove()
