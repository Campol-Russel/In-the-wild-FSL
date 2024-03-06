# In-The-Wild Filipino Sign Language Alphabet Dataset

Explore the Filipino Sign Language (FSL) image dataset. Capturing the richness of FSL through visual representations.

## Table of Contents
- [Description](#description)
- [Dataset Overview](#dataset-overview)
- [Abstract](#abstract)
- [Sample Visuals](#sample-visuals)
- [Dataset Access Request and Usage Guidelines](#dataset-access-request-and-usage-guidelines)
- [Contact Us](#contact-us)

## Project Website
Visit our [project website](https://campol-russel.github.io/In-the-wild-FSL/).

## Description
The objective of this study is to create and release a Filipino Sign Language (FSL) image dataset for public access. This comprehensive dataset includes 8100 images distributed across 27 classes representing the Filipino Alphabet, contributed by 100 participants who captured 81 images each using their devices. Users interested in utilizing this dataset are kindly requested to follow the provided guidelines and submit a form containing essential information, such as the representative's name, project details, how they intend to use the dataset, and their contact details. The dataset supervisor/s will review these requests to determine eligibility for dataset access. This process ensures that the dataset is used only for its intended purpose and serves as a record of authorized users, making any unauthorized usage easily detectable.

## Dataset Overview
- **Total Images:** 8100
- **Number of Classes:** 27
- **Collection Method:** Images captured by 100 participants using phone cameras
- **Dataset Structure:**
   - wildfsl_final_dataset
      - test
         - p7
            - image.jpg
            - image.xml
            - image2.jpg
            - image2.xml
            - …
         - p8
         - …
      - train
         - p0
            - image.jpg
            - image.xml
            - image2.jpg
            - image2.xml
            - …
         - p1
         - …
      - validation
         - P9
            - image.jpg
            - image.xml
            - image2.jpg
            - image2.xml
            - …
         - p81
         - …


## Abstract
In the Philippines, there is a scarcity of research on the deaf and mute culture, making institutions exclusive and limiting opportunities for the growth of the deaf and mute. This study aims to contribute to the development of real-time and accessible sign language detection and recognition systems. The dataset introduces a Filipino Sign Language dataset, addressing existing image dataset gaps. Data collection involved 100 adult participants capturing alphabet signs in varying environments. The open-source dataset contains 8,100 images, benchmarked in different models for broader use in the Filipino deaf community.

## Sample Visuals
![Sample Images of the Dataset](img/comp.png)

*Figure 1: Sample Visuals of the FSL Alphabet image dataset*

## Dataset Access Request and Usage Guidelines
Thank you for your interest in accessing the Filipino Sign Language (FSL) image dataset. To maintain the integrity and intended use of the dataset, we have established the following guidelines for access requests.

### Access Request Process:
1. **Submission of Request:**
   - State your interest in accessing the dataset.
   - Submit a request form with representative's name, project details, intended use, and contact details.
2. **Dataset Access Review:**
   - Supervisors will review requests to ensure eligibility and prevent misuse.
  
## Minimum Hardware Specs:
   - 64-bit processor and operating system
   - OS: Windows 10
   - Processor: Intel Core i7-4790K or AMD Ryzen 5 1500X
   - Memory: 8 GB RAM
   - Graphics: NVIDIA GeForce GTX 1080
   - Storage: 8 GB available space

## Applications to run:
   - Visual Studio Code or any source code editor.
   - Python
   - Google Collab

## API/Extensions/Libraties to run
### Image Classification Models
   1. torch - PyTorch deep learning framework.
   2. torch.nn - Module for constructing neural networks.
   3. torchvision.models - Pretrained models for computer vision.
   4. torchvision.transforms - Image transformations for data augmentation.
   5. torch.utils.data.DataLoader - DataLoader class for managing dataset loading in PyTorch.
   6. torch.utils.data.Dataset - Dataset class for creating custom datasets in PyTorch.
   7. tqdm - A fast, extensible progress bar for loops and pipelines.
   8. PIL.Image - The Python Imaging Library (PIL) module to open, manipulate, and save image files.
   9. os - Operating System dependent functionalities.
   10. xml.etree.ElementTree - A fast and simple XML API in Python.
   12. matplotlib.pyplot - A plotting library for the Python programming language and its numerical mathematics extension NumPy.
   13. numpy - A fundamental package for scientific computing with Python.
   14. torch.optim.lr_scheduler.MultiStepLR - Multi step learning rate scheduler.
   15. sklearn - The scikit-learn library for machine learning in Python.
   16. sklearn.model_selection.train_test_split - A method to split arrays or matrices into random train and test subsets.

### Object Detection Models
   1. Ultralytics - open-source deep learning libraries and tools used for the YOLOv8 Model
   2. Ultralytics YOLO API -  Imports the YOLO class from the Ultralytics package.
   3. IPython.display -  This module lets you display objects within a Jupyter Notebook, with the display function showing images inline.
   4. Roboflow Python SDK: The snippet imports the Roboflow Python SDK for accessing and managing datasets stored on Roboflow.

## Image Classification
A diverse set of established models was explored for image classification tasks, including AlexNet, VGGNet, ResNet, and DenseNet. These models were chosen for their proven effectiveness in image classification. The models were initially pre-trained on the ImageNet1K dataset, as recommended by PyTorch documentation, and then fine-tuned on the WildFSL dataset, allowing them to adapt to the specific challenges and nuances of the target tasks.

### Image Classification Models
#### Training
Training Models: The dataset can be used to train models for sign language detection and recognition. Use the provided XML files to extract the bounding box coordinates and class labels for each image.
Testing and Evaluation: You can use the dataset to evaluate the performance of pre-trained models or train your models and evaluate their accuracy on the dataset.   

To set the dataset path, find the file where the format is “[model]_new.py” and find the **train_dataset, valid_dataset, and test_dataset** variables, then change it to the path where you saved the dataset folders
   
#### Setting the Hyperparameters
To change the hyperparameters right below the path, you will see the variables for the hyperparameters. Change them to the hyperparameters as you see fit. For the benchmarking, the hyperparameters are set to the ones seen below. Take note that these hyperparameters are not the best parameters for the model these are just used for benchmarking purposes.
  
| Hyperparameter          | Variable Name    | Value      |
| ----------------------- | ---------------- | ---------- |
| Learning rate           | LR               | 0.001      |
| Learning rate scheduler | LR_sched         | [120, 150] |
| Initial epochs          | epochs           | 180        |
| Optimizer               | optimizer        | SGD        |
| Weight Decay            | weight_decay     | 0.0005     |
| Momentum                | momentum         | 0.9        |
| Input size              | tranforms.Resize | (224,224)  |
| Batch size              | batch_size       | 32         |
| Gamma                   | gamma            | 0.1        |

*Take note of where the weights of the model are saved; you will need the path to load the weights in validation/testing.*

#### Validation/Testing
To load the weights of the trained model, open the file “[model]_test”, find the variable **pretrained_weights_path**, and set it to where the weights of that model are saved.

After running, the accuracy will be displayed. Change the hyperparameters of the training file accordingly if you are validating, and to test, just run the file to check for accuracy.


## Object Detection
For object detection tasks, state-of-the-art models were incorporated, such as SSD (Single Shot Detector), YOLOv8 (You Only Look Once version 8), and SFDet (Single Fusion Detector). The experiment setups for these models were consistent with the classification models, focusing on the optimal hyperparameters for each model and the adjustments of epochs.
The primary metric of interest for classification models was accuracy, while for object detection models, the evaluation was based on the mean Average Precision (mAP).

### SFDET & SSD Training
#### Training
To load the dataset in the model, find the code ```parser.add_argument('--wildfsl_data_path', type=str,default='E:\\fsl dataset new\\dataset_resized', help='WildFSL dataset path')``` and change the path according to where the dataset is saved.

#### Setting the hyperparameters
For the SFDET and SSD hyperparameters, check the weights below and adjust the hyperparameters according to what is seen in the file.
[a link](https://github.com/user/repo/blob/branch/other_file.md)

### YOLOv8 Training
For YOLOv8, we utilized Google Collab, Ultralytics, and Roboflow

#### Training 
1. To load the dataset into Google Collab, we utilized Roboflow as a medium in connecting to Google Collab. We uploaded our data set to a Roboflow dataset and got the code snippet for the dataset. To access the code, you need to go to generate to generate a new version of the dataset or the version if you already have an existing one.
2. In the versions tab, click the custom train and upload the generated version.
3. Set the dropdown menu of the models to YOLOv8, then get the code snippet.
4. Paste the code snippet to your notebook, and after you run that cell, the dataset should be loaded into your notebook.
   
#### Setting the hyperparameters
To train and set the hyperparameters for the YOLOv8, use the code snippet below and change the hyperparameters as needed the values used for this run are just for benchmarking purposes and may not be the best hyperparameters for this model. 

| Hyperparameter          | Variable Name    | Value      |
| ----------------------- | ---------------- | ---------- |
| Batch size              | batch_size       | 32         |
| Epochs                  | epochs           | 180        |
| Learning rate           | LR               | 0.001      |
| Learning rate scheduler | LR_sched         | [120, 150] |
| Momentum                | momentum         | 0.9        |
| Weight Decay            | weight_decay     | 0.0005     |
| Learning rate Decay     | LR_sched         | 0.1        |
| Input size              | tranforms.Resize | 300        |
| Optimizer               | optimizer        | SGD        |
| Batch multiplier        | LR_sched         | 1          |
| (IoU) threshold         | gamma            | 0.5        |

Since Ultralyrics didn’t have the learning rate scheduler, we ran the training mode three times using the data from the first test and then change the learning rate according to the learning rate scheduler [120, 150], and the learning rate decay by 0.1 (Multiply the learning rate value by 0.1 when you reach epoch 120 and 150). See the codes below as a reference for using Ultralytics' training mode.

Epoch 1-119
- ```detect mode=train model=yolov8m.pt data=/content/extrdb-3/data.yaml epochs=119 imgsz=300 batch=32 lr0=0.001 weight_decay=0.0005 momentum=0.9 optimizer=SGD```

Epoch 120-149
- ```!yolo task=detect mode=train model=/content/runs/detect/train/weights/best.pt data=/content/extrdb-3/data.yaml epochs=30 imgsz=300 batch=32 lr0=0.0001 weight_decay=0.0005 momentum=0.9 optimizer=SGD```

Epoch 150-180
- ```!yolo task=detect mode=train model=/content/runs/detect/train2/weights/best.pt data={dataset.location}/data.yaml epochs=31 imgsz=300 batch=32 lr0=0.00001 weight_decay=0.0005 momentum=0.9 optimizer=SGD```

#### Validation/Testing
To test or validate, use the code snippet below
- ```!yolo task=detect mode=val model=/content/runs/detect/train3/weights/best.pt data={dataset.location}/data.yaml```

#### Reference Video
https://youtu.be/wuZtUMEiKWY

## Model Evaluation
The dataset has been tested on various models, including AlexNet, VGGNet, ResNet, DenseNet, SSD, YOLOv8, and SFDet. The hyperparameters used for this evaluation is seen in setting the hyperparameter part. The performance and best weights of each model are seen below

Classification Models

| Model            | Overall Accuracy | Link to weights |
| ---------------- | ---------------- | --------------- |
| AlexNet          | 72.6%            | link            |
| RestNet50        | 87.3%            | link            |
| VGG              | 90.1%            | link            |
| DenseNet         | 93.2%            | link            |

Object Detection Models

| Model            | Mean Average Precision  | Link to weights     |
| ---------------- | ----------------------- | ------------------- |
| SSD              | 96.1%                   | link                |
| SFDET-DenseNet   | 97.3%                   | link                |
| YOLOv8           | 98.8%                   | link                |


## Contact Us
If you have any questions, feel free to contact us at:

**Email:**
- arren.antioquia@dlsu.edu.ph
- campol_russel@dlsu.edu.ph
- carl_delacruz@dlsu.edu.ph
- jericho_dizon@dlsu.edu.ph


Built with [Pico](https://picocss.com) • [Source Code](https://github.com/picocss/examples/blob/master/v1-classless/index.html)
