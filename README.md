# Image Processing Script

This script performs image segmentation using advanced techniques, including **edge detection**, **morphological operations**, **K-means clustering**, and **object extraction**. The goal is to isolate objects from an image and place them on a custom gradient background.

---

## Setup Instructions

### **Clone the Repository**  
Run the following command in your terminal:  
```bash
git clone https://github.com/usergaia/IMAGE-PROC-PROJECT.git
cd IMAGE-PROC-PROJECT
```
2. **Ensure Dependencies**:
```bash
   - Install **MATLAB** or use **VSCode** with the MATLAB Extension.  
   - Install the **Statistics and Machine Learning Toolbox** in MATLAB.
  ```

4. **File Structure | Segmentation Accuracy Evaluation**:
```bash
   - Place input images in the **images/raw/** directory.  
   - Models are stored in the **models/** directory.
     To check accuracy:
   - Place Ground truth images in **images/ground_truth/**.
   - Place Ground truth masks in **images/ground_truth_mask/**.  
```
---

## Execution Workflow

### **Step 1: Read and Resize Image**  
- The input image is read and resized to **512x512 pixels** for standardization.  

### **Step 2: Edge Detection and Morphological Gradient**  
- Convert the image to grayscale and apply the **Canny edge detection** algorithm to identify edges.  
- Enhance edges using a **morphological gradient**:  
  Morphological Gradient = Dilation - Erosion  

### **Step 3: Create Initial Mask**  
- Fill edges to generate an initial mask.  
- Refine the mask using **morphological operations** (closing, opening, and hole filling).  
- Retain only the **largest object** in the mask.  

### **Step 4: Apply K-means Clustering**  
- Use **K-means clustering** on the masked region to separate the object from the background.  
- Update the mask based on the **larger cluster** (assumed to be the object).  

### **Step 5: Combine and Refine Masks**  
- Combine the **initial mask** and **K-means mask** for final refinement.  
- Perform additional steps like closing gaps, filling holes, and removing small objects.  
- Keep only the **largest connected component** in the final mask.  

### **Step 6: Extract Object**  
- Multiply the image with the final mask to isolate the object.  

### **Step 7: Generate Gradient Background**  
- Create a smooth **gradient background** using custom RGB channel values.  
- Replace the original background with the gradient.  

### **Step 8: Display Results**  
- Display the following images for comparison:  
  1. Original Image  
  2. Initial Mask  
  3. K-means Mask  
  4. Extracted Object  
  5. Final Result with Gradient Background  

### **Step 9: Future Work (Object Detection)**  
- **Placeholder** for integrating object detection functionality.  

---

## Requirements

- **MATLAB** or **VSCode** with MATLAB Extension.  
- **Statistics and Machine Learning Toolbox** (for K-means clustering).  
- Input images placed in the **images** directory.  
- Models stored in the **models** directory.  

---
