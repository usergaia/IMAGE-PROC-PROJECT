# 🖼️ Image Processing Script

## ✨ Features

This script utilizes several advanced techniques for precise image segmentation:
- **Edge Detection** - Identify object boundaries
- **Morphological Operations** - Refine object masks
- **K-means Clustering** - Separate objects from backgrounds
- **Object Extraction** - Isolate and process objects independently

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
│   ├── raw/              # Place input images here
│   ├── ground_truth/     # Ground truth images for accuracy evaluation
│   └── ground_truth_mask/# Ground truth masks
├── models/               # Stores segmentation models
└── scripts/              # Contains processing scripts
```

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

### Step 7: Generate Gradient Background
- Create a smooth **gradient background** using custom RGB channel values
- Replace the original background with the gradient

### Step 8: Display Results
The script displays the following images for comparison:
1. Original Image
2. Initial Mask
3. K-means Mask
4. Extracted Object
5. Final Result with Gradient Background

## 📊 Segmentation Accuracy Evaluation

To evaluate segmentation accuracy:
1. Place ground truth images in **images/ground_truth/**
2. Place ground truth masks in **images/ground_truth_mask/**
3. Run the evaluation script to compare results

## 📦 Requirements

- **MATLAB** or **VSCode** with MATLAB Extension
- **Statistics and Machine Learning Toolbox** (for K-means clustering)
