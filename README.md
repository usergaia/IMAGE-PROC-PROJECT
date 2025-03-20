# 🖼️ Image Processing Script

## ✨ Features

This script utilizes several advanced techniques for precise image segmentation:
- **Edge Detection** - Identify object boundaries
- **Morphological Operations** - Refine object masks
- **K-means Clustering** - Separate objects from backgrounds
- **Object Extraction** - Isolate and process objects independently
- **Comic Style Processing** - Apply artistic effects to foreground objects
- **Background Modification** - Create smooth gradient or blurred backgrounds
- **Object Detection & Classification** - Identify specific objects using machine learning

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
**Note**: To run the `gt_mask_converter`, ensure that the ground truth files and masks are in the same hierarchy as the `images` folder.  

## 📋 Execution Workflow

### Step 1: Read and Resize Image
- The input image is read and resized to **512×512 pixels** for standardization

### Step 2: Edge Detection and Morphological Gradient
- Convert the image to grayscale and apply the **Canny edge detection** algorithm
- Enhance edges using a **morphological gradient**:
  ```
  Morphological Gradient = Dilation - Erosion
  ```

### Step 3: Create Initial Mask
- Fill edges to generate an initial mask
- Refine using **morphological operations** (closing, opening, and hole filling)
- Retain only the **largest object** in the mask

### Step 4: Apply K-means Clustering
- Use **K-means clustering** on the masked region to separate object from background
- Convert image to LAB color space for better clustering results
- Update mask based on the **larger cluster** (assumed to be the object)

### Step 5: Combine and Refine Masks
- Combine **initial mask** and **K-means mask** for final refinement
- Perform additional refinements:
  - Close gaps
  - Fill holes
  - Remove small objects
- Keep only the **largest connected component** in the final mask

### Step 6: Extract Object
- Multiply the image with the final mask to isolate the object
- Also extract the background separately for further processing

### Step 7: Apply Styling
- **Comic Style Foreground Processing**:
  - Apply posterization for cartoon-like appearance
  - Boost colors for enhanced vibrancy
  - Increase saturation to make colors more vivid
  - Add warm tone to the foreground
  - Create and add black outlines for comic effect
- **Background Processing**:
  - Apply Gaussian blur to the background for depth effect

### Step 8: Display Results
The script displays the following images for comparison:
1. Original Image
2. Each step of the segmentation process
3. The extracted foreground and background
4. The final composite image with artistic styling

### Step 9: Object Detection and Classification
- Extract features from the segmented object
- Use a pre-trained KNN model to identify common objects:
  - Burj Khalifa
  - Basketball
  - Car
  - Clown Fish
  - Logitech Mouse
- Display bounding boxes and object labels on the final result

### Step 10: Performance Evaluation 
- Calculate **Intersection over Union (IoU)** between the predicted mask and ground truth
- Display a comparison between expected and actual extraction results

## 📊 Segmentation Accuracy Evaluation

To evaluate segmentation accuracy:
1. Place ground truth images in **images/ground_truth/**
2. Place ground truth masks in **images/ground_truth_mask/**
3. The script automatically calculates and displays **IoU (Intersection over Union)** accuracy
4. Visual comparison of expected vs. actual extraction is displayed

## 📦 Requirements

- **MATLAB** or **VSCode** with MATLAB Extension
- **Statistics and Machine Learning Toolbox** (for K-means clustering and KNN classification)
- **Image Processing Toolbox** (for advanced morphological operations)
