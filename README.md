# Face Recognition from Features

## Task:
 Explore the dataset and extract hand-crafted features such as local binary pattern,Gabor filter, Laplacian of Gaussian (LoG), and Gray-Level Co-occurrence Matrix(GLCM). Combine these features and apply standard machine learning models to thefeature set. The classifier aims to detect persons accurately. Report the results in detail.

 ## Dataset:
 Yale feature dataset

## Goal of the project:
to develop a system that can recognize which subject an image belongs to, based on the features extracted from the images and applying machine learning models.

## Understanding the Task:
Given a dataset with images of different subjects under various conditions (like different facial expressions and lighting). The task is to develop a system that can accurately identify the subject in any given image. For example, if an image of subject14 is fed under any condition (like subject14.surprised), the system should correctly identify it as subject14.

# Face Recognition Pipeline in main.py

## 1. GITHUB CLONE
Clone the GitHub repository that contains the dataset for easy implementation on Google Colab. This also contains the extracted features to avoid calculating them in every runtime.  
If calculating Feature Vectors again is not needed, after cloning, one can go to the **CLASSIFICATION** section directly and start by combining the features.

---

## 2. Importing Libraries
Run this block to import all the necessary libraries.

---

## 3. Visualizing Sample Data
This visualizes one of the images from the dataset.

---

## 4. Convert to `.npy`
Run this block to convert all the data (in GIF format) to ndarray.

---

## 5. Face Detection
This block uses the Haar Cascade Classifier to detect faces in the images. The faces are then cropped and prepared for further processing.

---

## 6. Image Visualization and Preprocessing

### This section includes:
- **Look at the data**
- **Distribution of images and expressions**
- **Check if all of them are of the same size**
- **Normalize**
- **Resize**
- **Show Image Histogram**

### Subsections:
- **Data Visualization**  
  This part loads and visualizes a grid of images corresponding to different emotions for a given subject ID from `.npy` files in the cropped data directory, displaying them in grayscale.

- **Distribution of Data**  
  This part counts the number of images for each facial expression in the dataset and visualizes the class distribution as a bar chart.

- **Check if all the images are of the same size**  
  This part checks if the images in the data before and after cropping are the same size.

- **Normalization**  
  This normalizes the cropped data.

- **Resizing**  
  This resizes the cropped data to `(128, 128)`.

- **Histogram of Pixel Values of the Image**  
  This shows the Image Histogram for original and normalized images.

---

## 7. Feature Extraction

### 7.1 Local Binary Pattern
- **Code:**  
  This part finds the Local Binary Pattern array and Histogram.  

- **Visualizing:**  
  This part visualizes the LBP image and corresponding Histogram, then plots histograms of LBP of textures by setting thresholding.

### 7.2 Gabor Features
- **Code:**  
  This part first makes the Bank of Gabor Kernels and then extracts the Gabor features and saves them in a directory.

- **Visualizing:**  
  This part visualizes the images after convolution with the Gabor filters.

### 7.3 LoG (Laplacian of Gaussian)
- **Code:**  
  This part calculates the Laplacian of Gaussian of each image and saves them.

- **Visualizing:**  
  This part visualizes the images after applying the Laplacian of Gaussian operation.

### 7.4 GLCM (Gray Level Co-occurrence Matrix)
- **Code:**  
  This part calculates the GLCM of each image and saves them.

- **Visualizing:**  
  This part visualizes the effect of different GLCM properties (contrast, dissimilarity,
 homogeneity, energy, and correlation) on the image by applying transformations based on
 each property and displaying the manipulated images alongside the original and the
 heatmaps of the individual features for an image.

---

## 8. Classification

### 8.1 Combining Features
This part combines all the 4 feature vectors and creates a feature vector and corresponding labels.

### 8.2 Splitting Train and Test Data
This part splits the data into 80% training and 20% testing data.

### 8.3 Scaling and Applying PCA
This part scales the data using `StandardScaler()` and then applies PCA to the data.

### 8.4 Random Forest Classifier
This part applies the Random Forest Classifier on:
- Original data
- Scaled data
- Data obtained after applying PCA

### 8.5 Logistic Regression
This part applies Logistic Regression on:
- Original data
- Scaled data
- Data obtained after applying PCA

### 8.6 Support Vector Classifier (SVC)
This part applies SVC on:
- Scaled data
- Data obtained after applying PCA
