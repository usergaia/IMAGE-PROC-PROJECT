# 🖼️ Image Processing Script

## ✨ Features

This script utilizes several advanced techniques for precise image segmentation and artistic processing:
- **Edge Detection** - Identify object boundaries
- **Morphological Operations** - Refine object masks
- **K-means Clustering** - Separate objects from backgrounds
- **Object Extraction** - Isolate and process objects independently
- **Comic Style Processing** - Apply artistic effects to foreground objects
- **Background Modification** - Apply Gaussian blur to background
- **Object Detection & Classification** - Identify specific objects using machine learning
- **Performance Evaluation** - Calculate IoU accuracy with ground truth

## 🚀 Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/usergaia/IMAGE-PROC-PROJECT.git
cd IMAGE-PROC-PROJECT
```

### Dependencies

- **MATLAB** or **VSCode** with the MATLAB Extension
- **Statistics and Machine Learning Toolbox** in MATLAB

### File Structure

```
IMAGE-PROC-PROJECT/
├── images/
│   ├── raw/               # Place input images here
│   ├── ground_truth/      # Ground truth images for accuracy evaluation
│   └── ground_truth_mask/ # Ground truth masks
├── models/                # Stores segmentation models
└── src/                   # Contains processing scripts
└── main.m
```
**Note**: To run the `gt_mask_converter`, ensure that this is in the same hierarchy as the `images` folder.  

## 📋 Execution Workflow

### Step 1: Read and Resize Image
- The input image is read from the 'images/raw/' directory
- Ground truth images and masks are loaded for later accuracy evaluation
- All images are resized to **512×512 pixels** for standardization

### Step 2: Apply Canny Edge Detection and Morphological Gradient
- Convert the image to grayscale for edge detection
- Apply the **Canny edge detection** algorithm with thresholds [0.04, 0.15]
- Enhance edges using a **morphological gradient** with a disk structuring element of radius 2

### Step 3: Create Initial Mask and Reduce Noise with Morphological Operations
- Fill holes in the edges to generate an initial mask
- Refine using **morphological operations**:
  - Apply closing with disk(5) to connect nearby edges
  - Apply opening with disk(2) to remove small noise
  - Remove small disconnected regions (<500 pixels)
- Keep only the **largest connected component** if multiple objects are detected

### Step 4: Apply K-means Only on the Masked Region with the Color Space LAB
- Convert image to LAB color space for better color-based segmentation
- Extract pixels within the initial mask for clustering
- Apply K-means clustering (k=2) only on the masked region
- Determine the main cluster (typically the larger one) to represent the object
- Create a K-means based mask by selecting pixels from the main cluster

### Step 5: Refine K-means Segmentation with Morphological Operations
- Apply closing with disk(5) to connect nearby parts
- Fill any remaining holes in the mask
- Remove small objects (<500 pixels)
- Keep only the **largest connected component** in the final mask
- Fall back to initial mask if no components remain after refinement

### Step 6: Extract Objects Using Final Mask
- Extract foreground object by multiplying the image with the final mask
- Extract background by multiplying the image with the inverse of the final mask

### Extra: Modify the Extracted Objects
#### Foreground - Comic Style Processing
- Apply posterization (8 levels) for cartoon-like appearance
- Boost colors by a factor of 5 to enhance vibrancy
- Increase saturation by 40% in HSV color space
- Add a warm tone to the foreground with factors [1.1, 1.0, 0.85]
- Create black outlines using Canny edge detection and dilation
- Apply the comic style only to the masked foreground area

#### Background - Gaussian Blur
- Apply Gaussian blur (sigma=10) to the background for depth effect

#### Combine Foreground and Background
- Start with the blurred background
- Add the comic-styled foreground on top

### Step 7: Display Results
The script creates a visualization showing:
- Original Image
- Each intermediate step of the segmentation process (8 steps)
- Processed background and foreground components
- Final combined output with artistic styling

### Step 8: Object Detection on Segmented Object in the Image
- Extract region properties (BoundingBox) from the final mask
- Load a pre-trained KNN model from 'models/knnModel.mat'
- Extract image features using the `extractImageFeatures` function:
  - Standardize image size to 256×256
  - Calculate texture features using Gray Level Co-occurrence Matrix (GLCM)
  - Extract Haralick features (energy, contrast, homogeneity, entropy)
  - Compute normalized color histograms for each RGB channel
- Predict the object class from 5 categories:
  - Burj Khalifa
  - Basketball
  - Car
  - Clown Fish
  - Logitech Mouse
- Draw bounding boxes and labels on the final image

### Step 9: Performance Evaluation (Segmentation IoU)
- Preprocess the ground truth mask for comparison
- Ensure both predicted and ground truth masks are binary
- Calculate Intersection over Union (IoU):
  ```
  IoU = (Intersection / Union) * 100%
  ```
- Display IoU accuracy percentage
- Create a visual comparison showing:
  - Expected extracted object (ground truth)
  - Actual extracted object
  - Binary ground truth mask
  - Binary predicted mask

## 📊 Segmentation Accuracy Evaluation

The script automatically evaluates segmentation accuracy using:
- **Intersection over Union (IoU)** - A standard metric for segmentation quality
- Visual comparison between expected and actual results
- Console output showing exact IoU percentage

## 📦 Requirements

- **MATLAB** or **VSCode** with MATLAB Extension
- **Statistics and Machine Learning Toolbox** (for K-means clustering and KNN classification)
- **Image Processing Toolbox** (for advanced morphological operations and texture feature extraction)
