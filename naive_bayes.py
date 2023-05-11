'''import numpy as np
import cv2
import os
from skimage.feature import greycomatrix, greycoprops

# Load images from a directory
img_folder = 'F:/Research Work/L/lung'
images = []
for filename in os.listdir(img_folder):
    img = cv2.imread(os.path.join(img_folder, filename), 0) # Read as grayscale
    images.append(img)
    

# Apply contrast enhancement    
enhanced_images = []
for img in images:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    enhanced_images.append(enhanced_img)
    
# Apply noise reduction
denoised_images = []
for enhanced_img in images:
    denoised_img = cv2.fastNlMeansDenoising(enhanced_img, None, h=10, templateWindowSize=7, searchWindowSize=21)
    denoised_images.append(denoised_img)
    
# Apply Gaussian blur
blurred_images = []
for denoised_img in images:
    blurred_img = cv2.GaussianBlur(denoised_img, (5, 5), 0) # Kernel size (5, 5) can be adjusted
    blurred_images.append(blurred_img)


# Convert images to binary format
binary_images = []
for blurred_img in images:
    binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_images.append(binary_img)
    
# Define structuring element (periodic line)
ksize = 21 # Kernel size can be adjusted
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
periodic_line = np.tile(kernel, (img.shape[0] // ksize + 1, 1))[:img.shape[0], :]

# Apply morphological opening with periodic line
opened_images = []
for binary_img in images:
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, periodic_line)
    opened_images.append(opened_img)


# Image normalization
normalized_images = []
for opened_img in images:
    normalized_img = cv2.normalize(opened_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_images.append(normalized_img)

# Resize images
resized_images = []
for opened_img in normalized_images:
    resized_img = cv2.resize(opened_img, (224, 224)) # Resize to 224x224
    resized_images.append(resized_img)

# Extract GLCM features
glcm_features = []
for resized_img in images:
    glcm = greycomatrix(resized_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True)
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    glcm_features.append([contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0], ASM[0,0]])

# Extract statistical features
import math as m
stat_features = []
for resized_img in images:
    mean = np.mean(resized_img)
    std = np.std(resized_img)
    skewness = np.mean(((resized_img - mean)/std)**3)
    kurtosis = np.mean(((resized_img - mean)/std)**4) - 3
    fifth_central_mo = np.mean(((resized_img - mean)/std)**5)
    Sixth_central_mo = np.mean(((resized_img - mean)/std)**6)
    RSM = m.sqrt((resized_img.mean())**2)
    stat_features.append([mean, std, skewness, kurtosis,fifth_central_mo,Sixth_central_mo,RSM])

# Combine features and labels
X = np.hstack((glcm_features, stat_features))
y = np.array([0 if 'non-cancer' in f else 1 for f in os.listdir(img_folder)])

# Shuffle the data
idx = np.random.permutation(X.shape[0])
X = X[idx]
y = y[idx]   
'''


from PIL import Image, ImageEnhance
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2
from skimage.feature import greycomatrix, greycoprops



# Load images from a directory
img_folder = 'F:/Research Work/L/lung'
images = []
for filename in os.listdir(img_folder):
    img = Image.open(os.path.join(img_folder, filename)).convert('L') # Read as grayscale
    images.append(img)

# Apply contrast enhancement
enhanced_images = []
for img2 in images:
    enhancer = ImageEnhance.Contrast(img2)
    enhanced_img = enhancer.enhance(2.0) # Increase contrast
    enhanced_images.append(enhanced_img)

# Apply noise detection
denoised_images = []
for enhanced_img in enhanced_images:
    denoised_img = enhanced_img.filter(ImageFilter.MedianFilter(size=3))
    denoised_images.append(denoised_img)
    
# Apply Gaussian blur
blurred_images = []
for denoised_img in denoised_images:
    blurred_img = denoised_img.filter(ImageFilter.GaussianBlur(radius=5)) # Radius can be adjusted
    blurred_images.append(blurred_img)
    
# Convert images to binary format
binary_images = []
for blurred_img in blurred_images:
    img_array = np.array(blurred_img)
    _, binary_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img = Image.fromarray(binary_img.astype(np.uint8))
    binary_images.append(binary_img)
    
# Define structuring element (periodic line)
ksize = 21 # Kernel size can be adjusted
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))

# Apply periodic line structuring element
periodic_line_images = []
for binary_img in binary_images:
    img_array = np.array(binary_img)
    periodic_line = np.tile(kernel, (img_array.shape[0] // ksize + 1, 1))[:img_array.shape[0], :]
    eroded_img = cv2.erode(img_array, periodic_line, iterations=1)
    periodic_line_img = Image.fromarray(eroded_img.astype(np.uint8))
    periodic_line_images.append(periodic_line_img)
    
# Image normalization
normalized_images = []
for opened_img in images:
    img_array = np.array(opened_img)
    normalized_img = cv2.normalize(img_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    normalized_img = Image.fromarray(normalized_img)
    normalized_images.append(normalized_img)


# Set the desired size for the resized images
new_size = (224, 224)

# Resize images
resized_images = []
for filename in os.listdir(img_folder):
    img_path = os.path.join(img_folder, filename)
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, new_size)
    resized_images.append(resized_img)
    
# Extract GLCM features for each image in the directory
glcm_features = []
for img_path in os.listdir(img_folder):
    # Read image and convert to grayscale
    img = cv2.imread(os.path.join(img_folder, img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize image to 224x224
    resized = cv2.resize(gray, (224, 224))
    
    # Calculate GLCM features
    glcm = greycomatrix(resized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True)
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    
    # Append features to list
    glcm_features.append([contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0], ASM[0,0]])


import math as m    
# Create an empty list to store the feature vectors
stat_features = []

# Loop over each image in the directory
for filename in os.listdir(img_folder):
    # Load the image and convert it to grayscale
    img = cv2.imread(os.path.join(img_folder, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the statistical features
    mean = np.mean(gray)
    std = np.std(gray)
    skewness = np.mean(((gray - mean)/std)**3)
    kurtosis = np.mean(((gray - mean)/std)**4) - 3
    fifth_central_mo = np.mean(((gray - mean)/std)**5)
    sixth_central_mo = np.mean(((gray - mean)/std)**6)
    RSM = m.sqrt((gray.mean())**2)
    
    # Add the feature vector to the list
    stat_features.append([mean, std, skewness, kurtosis, fifth_central_mo, sixth_central_mo, RSM])
    

# Create an empty list to store the feature vectors
shape_features = []

# Loop over each image in the directory
for filename in os.listdir(img_folder):
    # Load the image and convert it to grayscale
    img = cv2.imread(os.path.join(img_folder, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to get a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over each contour
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate the compactness of the contour
        if area > 0:
            compactness = (perimeter**2) / (4 * np.pi * area)
        else:
            compactness = 0
        
        # Add the feature vector to the list
        shape_features.append([area, perimeter, compactness])
        
        
import os
import cv2
import numpy as np

# Create an empty list to store the feature vectors
fractal_features = []

# Loop over each file in the directory
for file in os.scandir(img_folder):
    # Check if the file is an image
    if file.name.endswith(".jpg") or file.name.endswith(".png"):
        # Load the image
        img = cv2.imread(file.path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to get a binary image
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Compute the box-counting dimension
        N = []
        S = []
        for s in range(1, 10):
            scale = 2**s
            s_inv = 1.0 / scale
            count = 0
            for i in range(scale):
                for j in range(scale):
                    roi = thresh[int(i*s_inv*img.shape[0]):int((i+1)*s_inv*img.shape[0]), int(j*s_inv*img.shape[1]):int((j+1)*s_inv*img.shape[1])]
                    if np.sum(roi) > 0:
                        count += 1
            N.append(count)
            S.append(scale)

        log_N = np.log(N)
        log_S = np.log(S)
        p = np.polyfit(log_S, log_N, 1)

        # Add the feature vector to the list
        fractal_features.append([p[0]])



# Combine all features
shape_features_resized = shape_features[:len(glcm_features)]
fractal_features_resized = fractal_features[:len(glcm_features)]
if len(fractal_features_resized) == 1:
    fractal_features_resized = np.repeat(fractal_features_resized, len(glcm_features))
fractal_features_resized = np.reshape(fractal_features_resized, (len(glcm_features), 1))
X = np.hstack((glcm_features, stat_features, shape_features_resized, fractal_features_resized))
y = np.array([0 if 'non-cancer' in f else 1 for f in os.listdir(img_folder)])




# Shuffle the data
idx = np.random.permutation(X.shape[0])
X = X[idx]
y = y[idx]



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data (e.g. using the code from previous questions)
#X = np.load('path/to/preprocessed/X.npy')
#y = np.load('path/to/preprocessed/y.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Instantiate the Naive Bayes classifier
nb = GaussianNB()

# Fit the model on the training data
nb.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = nb.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
#print("Accuracy:", acc)
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall *100))
print("F1 score: {:.2f}%".format(f1*100))


from sklearn.metrics import roc_curve, auc

# Train an decision tree on the training data
#clf = SVC(kernel='linear', probability=True)
#clf.fit(X_train, y_train)


# Instantiate the Naive Bayes classifier
nb = GaussianNB()

# Fit the model on the training data
nb.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = nb.predict(X_test)

# Compute the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes')
plt.legend(loc="lower right")
plt.show()