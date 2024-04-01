import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from xml.etree import ElementTree as ET
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.scorecam import ScoreCam as ScoreCAMForAlexVGG
from src.scorecam_dense_res import ScoreCAM as ScoreCAMForDenseRes

def execute(model_type, mode, save_path, train_path, valid_path, test_path, LR, LR_sched, epochs, optimizer, weight_decay, momentum, input_size, pretrained_weights_path=None, scorecam=False):
    """Trains or tests a neural network model for image classification, with optional ScoreCAM analysis.

    Args:
        model_type (str): Type of neural network model to use ('resnet', 'densenet', 'vgg', or 'alexnet').
        mode (str): Mode of operation ('train' or 'test').
        save_path (str): Path to save trained model and weights.
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the validation dataset.
        test_path (str): Path to the test dataset.
        LR (float): Learning rate for the optimizer.
        LR_sched (list of int): Milestones for the learning rate scheduler.
        epochs (int): Number of epochs for training.
        optimizer (str): Optimizer to use ('sgd' or other).
        weight_decay (float): Weight decay parameter for the optimizer.
        momentum (float): Momentum parameter for SGD optimizer.
        input_size (tuple of int): Size of input images (height, width).
        pretrained_weights_path (str, optional): Path to pre-trained weights for testing.
        scorecam (bool, optional): Whether to run ScoreCAM analysis during testing. Defaults to False.

    Raises:
        ValueError: If an unsupported model type or optimizer is provided, or an invalid mode is specified.

    Returns:
        None
    """

    # Transformations
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    # Create datasets and data loaders
    train_dataset = CustomDataset(root_dir=train_path, transform=transform)
    valid_dataset = CustomDataset(root_dir=valid_path, transform=transform)
    test_dataset = CustomDataset(root_dir=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = 27 # WildFSL classes

    # Initialize the model
    if model_type == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, num_classes)
    elif model_type == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_type == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    else:
        raise ValueError("Unsupported model type")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_sched, gamma=0.1)

    if mode == 'train':
        # Training loop
        for epoch in range(epochs):
            model.train()
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs} - Training') as pbar:
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    pbar.update(1)

            # Validation
            model.eval()
            with torch.no_grad():
                total = 0
                correct = 0
                with tqdm(total=len(valid_loader), desc=f'Epoch {epoch}/{epochs} - Validation') as pbar:
                    for inputs, labels in valid_loader:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        pbar.update(1)

            accuracy = correct / total
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}, Validation Accuracy: {accuracy}')

            # Save model and weights, adjust rate of saving if needed
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), os.path.join(save_path, f'{model_type}_epoch_{epoch + 1}.pth'))

            scheduler.step()

        # Save the final model
        torch.save(model.state_dict(), os.path.join(save_path, f'{model_type}_final.pth'))

    elif mode == 'test':
        if pretrained_weights_path is None:
            raise ValueError("Please provide path to pre-trained weights for testing")

        # Load pre-trained weights
        if os.path.exists(pretrained_weights_path):
            model.load_state_dict(torch.load(pretrained_weights_path))
        else:
            raise FileNotFoundError(f"Pre-trained weights not found at {pretrained_weights_path}")

        # Test the model
        model.eval()
        with torch.no_grad():
            if scorecam:
                # Initialization of ScoreCAM based on the model type
                if model_type in ['resnet', 'densenet']:
                    target_layer = model.layer4[-1] if model_type == 'resnet' else model.features.norm5
                    scorecam = ScoreCAMForDenseRes(model, target_layer)
                elif model_type in ['alexnet', 'vgg']:
                    target_layer = 10 if model_type == 'alexnet' else 29 # Define the appropriate target layer for AlexNet and VGGNet
                    scorecam = ScoreCAMForAlexVGG(model, target_layer)
                else:
                    raise ValueError("Unsupported model type")
            else:
                print("Scorecam is disabled, now testing..")

            total = 0
            correct = 0
            class_correct = [0] * num_classes
            class_total = [0] * num_classes
            with tqdm(total=len(test_loader), desc='Testing') as pbar:
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Update per-class accuracy
                    for i in range(num_classes):
                        label_mask = labels == i
                        class_total[i] += label_mask.sum().item()
                        class_correct[i] += (predicted[label_mask] == labels[label_mask]).sum().item()

                        # Generate overlay for each class
                        if scorecam:
                            overlay = generate_overlay(inputs[label_mask], model, scorecam, i, label_mask, save_path)
                            if overlay is not None:
                                cv2.imwrite(f"class_{i}_overlay.png", overlay)
                                print(f"class_{i}_overlay.png generated")
                        # else:                                                 debug purposes
                        #     print(f"Overlay for class {i} is empty")

                    pbar.update(1)

        test_accuracy = correct / total
        print(f'Test Accuracy: {test_accuracy}')

        # Calculate and print per-class accuracy
        for i in range(num_classes):
            class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f'Accuracy for class {i}: {class_acc}')

def generate_overlay(inputs, model, scorecam, class_idx, label_mask, save_path, use_scorecam=False):
    """
    Generates and saves an overlay image using ScoreCAM-generated heatmaps, if enabled.
    
    This function creates a heatmap for a given class index using the ScoreCAM technique,
    overlays it on the original input images, and saves the resulting images to disk.
    The operation is performed only if the use_scorecam flag is set to True.

    Args:
        inputs (torch.Tensor): Input images in a batch, as a tensor.
        model (torch.nn.Module): The neural network model being analyzed.
        scorecam (ScoreCAMForAlexVGG or ScoreCAMForDenseRes): An instance of ScoreCAM tailored to the model architecture.
        class_idx (int): The index of the class for which the heatmap is generated.
        label_mask (torch.Tensor): A boolean mask tensor indicating the presence of the target class in each image of the batch.
        save_path (str): The directory path where the overlay images will be saved.
        use_scorecam (bool, optional): Flag indicating whether to perform ScoreCAM analysis. Defaults to False.
    
    Returns:
        np.array: The generated overlay image as a NumPy array. Returns None if ScoreCAM is not used or if there's no input for the specified class.
    
    Raises:
        FileNotFoundError: If the save_path directory does not exist and cannot be created.
        ValueError: If there are issues generating the heatmap or overlay (typically related to input tensor dimensions or types).
    """
    
    # Check if label_mask selects at least one element
    if torch.sum(label_mask) == 0:
        return None  # Return None if no elements are selected

    # Get the heatmap
    heatmap = scorecam.generate_cam(input_image=inputs, target_class=class_idx)

    # Extract numpy array from PyTorch tensor and convert to OpenCV image format
    image_np = inputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Convert to PIL image
    pil_img = Image.fromarray(image_np[0])

    # Resize heatmap to match the size of the original image
    heatmap_resized = cv2.resize(heatmap, (pil_img.size[0], pil_img.size[1]))

    # Normalize heatmap values to range [0, 255] and convert to uint8
    heatmap_resized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(image_np[0], 0.7, heatmap_colored, 0.3, 0)

    # Create a directory for the class if it doesn't exist
    class_dir = os.path.join(save_path, f"class_{class_idx}")
    os.makedirs(class_dir, exist_ok=True)

    # Save overlay image with a unique filename
    num_images = len(os.listdir(class_dir))
    overlay_path = os.path.join(class_dir, f"generated_image_{num_images + 1}.png")
    cv2.imwrite(overlay_path, overlay)

    return Overlay
    
class CustomDataset(Dataset):
    """Custom dataset class for loading images and their corresponding labels.

    Args:
        root_dir (str): Root directory containing the dataset.
        transform (callable, optional): Optional transform to be applied to the images.

    Attributes:
        root_dir (str): Root directory containing the dataset.
        transform (callable): Optional transform to be applied to the images.
        image_files (list of str): List of paths to image files.
        labels (list of str): List of corresponding labels.
    """

    def __init__(self, root_dir, transform=None):
        """Initializes the CustomDataset class.

        Args:
            root_dir (str): Root directory containing the dataset.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        for subfolder in subfolders:
            for filename in os.listdir(subfolder):
                if filename.endswith('.jpg'):
                    self.image_files.append(os.path.join(subfolder, filename))
                    
                    # Read the XML file to get the label
                    xml_filename = os.path.join(subfolder, filename.replace('.jpg', '.xml'))
                    label = self._parse_voc_xml(xml_filename)
                    self.labels.append(label)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Gets the idx-th sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')

        # Assuming self.labels contains string labels
        label_str = self.labels[idx]
        
        # Convert string label to numerical index
        label_num = self.label_to_index(label_str)

        label = torch.tensor(label_num, dtype=torch.long)  # Convert label to tensor with long dtype

        if self.transform:
            image = self.transform(image)

        return image, label

    def label_to_index(self, label_str):
        """Converts a string label to a numerical index.

        Args:
            label_str (str): String representation of the label.

        Returns:
            int: Numerical index corresponding to the label.
        """
        # Mapping from string labels to numerical indices
        label_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
                         'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
                         't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, '0': 26}

        return label_mapping[label_str]

    def _parse_voc_xml(self, xml_path):
        """Parses a PASCAL VOC XML file to extract the label.

        Args:
            xml_path (str): Path to the XML file.

        Returns:
            str: Extracted label from the XML file.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        label = root.find('object').find('name').text
        return label

# sample only, change values based on your parameters
if __name__ == "__main__":
    execute(model_type='alexnet', 
         mode='test', 
         save_path="E:\\repos\\pytorch-cnn-visualizations\\overlays", 
         train_path="E:\\onedrive in E\\OneDrive - De La Salle University - Manila\\Dataset\\In-the-wild FSL Alphabet Dataset.v6i.voc\\wildfsl_final_dataset\\train",
         valid_path="E:\\onedrive in E\\OneDrive - De La Salle University - Manila\\Dataset\\In-the-wild FSL Alphabet Dataset.v6i.voc\\wildfsl_final_dataset\\validation",
         test_path="E:\\onedrive in E\\OneDrive - De La Salle University - Manila\\Dataset\\In-the-wild FSL Alphabet Dataset.v6i.voc\\wildfsl_final_dataset\\test",
         LR=0.001,
         LR_sched=[120, 150],
         epochs=180,
         optimizer="sgd",
         weight_decay=0.1,
         momentum=0.9,
         input_size=(224,224),
         pretrained_weights_path="E:\\onedrive in E\\OneDrive - De La Salle University - Manila\\Dataset\\In-the-wild FSL Alphabet Dataset.v6i.voc\\alexnet run in hg - carl\\weights\\alexnet_epoch_85.pth",
         scorecam=True)
