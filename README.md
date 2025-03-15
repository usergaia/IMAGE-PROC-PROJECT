# IMAGE-PROC-PROJECT

# Project Timeline and Tasks

## Day 1: Planning and Data Preparation
### Both Members:
- **Understand the project requirements:** Discuss the goals, deliverables, and expected outputs.
- **Set up MATLAB:** Ensure both members have MATLAB installed with the Image Processing Toolbox.
- **Collect images:** Gather at least 5 images (e.g., animals, landscapes, objects) and ensure they are diverse.
- **Preprocess images:** Resize all images to a consistent resolution (e.g., 512x512) using MATLAB's `imresize()` function.
- **Divide tasks:** Assign specific responsibilities for the next few days.

---

## Day 2-3: Image Segmentation
### Member 1: Color Segmentation
- **Convert images to different color spaces:**
  - Use MATLAB functions like `rgb2hsv()`, `rgb2ycbcr()` to convert RGB images to HSV and YCbCr.
- **Apply color thresholds:**
  - Use `imbinarize()` or create custom thresholds to isolate objects based on color.
  - Example: Extract green regions from an image using HSV space.
- **Extract region properties:**
  - Use `regionprops()` to analyze properties like area, centroid, and bounding boxes.

### Member 2: Edge Detection
- **Apply edge detection:**
  - Use MATLAB functions like `edge()` with Sobel, Canny, or Prewitt methods.
  - Example: `edge(image, 'Canny')`.
- **Enhance edges:**
  - Use morphological operations like `imdilate()` and `imerode()` to refine edges.
  - Example: Dilate edges to make them more visible.

---

## Day 4: Clustering Segmentation
### Member 1: K-Means Clustering
- **Implement K-means clustering:**
  - Use `kmeans()` to cluster pixels based on intensity or color.
  - Example: Cluster the image into 3-4 regions.
- **Remove noise:**
  - Use `bwareaopen()` to remove small noisy regions from the segmented output.

### Member 2: Assist and Prepare for Object Detection
- **Review clustering results:** Help Member 1 refine the clustering output.
- **Start object detection preparation:**
  - Research MATLAB functions for connected component analysis (`bwconncomp()`) and blob detection.

---

## Day 5: Object Detection
### Member 1: Connected Component Analysis
- **Label distinct objects:**
  - Use `bwconncomp()` or `bwlabel()` to label connected components in the binary image.
- **Calculate object properties:**
  - Use `regionprops()` to calculate centroid, area, and bounding boxes for each object.

### Member 2: Bounding Boxes and Visualization
- **Draw bounding boxes:**
  - Use `rectangle()` to draw bounding boxes around detected objects.
- **Overlay results:**
  - Overlay segmentation results on the original image using `imoverlay()` or custom code.

---

## Day 6: Scene Classification
### Member 1: Feature Extraction
- **Extract color histograms:**
  - Use `imhist()` to compute color histograms for each image.
- **Extract texture features:**
  - Use `graycomatrix()` and `graycoprops()` to compute GLCM features (e.g., contrast, correlation).

### Member 2: Train Classifier
- **Prepare training data:**
  - Create a feature matrix (color histograms + texture features) and label vector.
- **Train SVM or KNN:**
  - Use `fitcsvm()` for SVM or `fitcknn()` for KNN in MATLAB.
  - Example: Train a classifier to distinguish between "indoor" and "outdoor" scenes.

---

## Day 7: Visualization and Integration
### Member 1: Final Segmentation Overlay
- **Overlay segmentation results:**
  - Combine color segmentation, edge detection, and clustering results into a single output.
  - Use `imfuse()` or custom code to overlay results on the original image.

### Member 2: Integrate Object Detection and Classification
- **Integrate object detection:**
  - Ensure bounding boxes and object properties are displayed correctly.
- **Test scene classification:**
  - Run the trained classifier on test images and display the predicted scene labels.

---

## Day 8-9: Finalization and Testing
### Both Members:
- **Test the entire pipeline:**
  - Run the pipeline on all 5 images and verify the results.
- **Debug and refine:**
  - Fix any issues with segmentation, object detection, or classification.
- **Prepare deliverables:**
  - Create a presentation or report summarizing:
    - Methodology (steps taken for segmentation, detection, and classification).
    - Results (visual outputs for each image).
    - Challenges faced and how they were resolved.

### Optional:
- If time permits, improve the pipeline (e.g., better thresholds, more accurate classifier).

---

## Tools and MATLAB Functions to Use
- **Image Preprocessing:** `imresize()`, `imread()`, `imshow()`
- **Color Segmentation:** `rgb2hsv()`, `rgb2ycbcr()`, `imbinarize()`, `regionprops()`
- **Edge Detection:** `edge()`, `imdilate()`, `imerode()`
- **Clustering:** `kmeans()`, `bwareaopen()`
- **Object Detection:** `bwconncomp()`, `bwlabel()`, `regionprops()`, `rectangle()`
- **Feature Extraction:** `imhist()`, `graycomatrix()`, `graycoprops()`
- **Classification:** `fitcsvm()`, `fitcknn()`
- **Visualization:** `imoverlay()`, `imfuse()`
