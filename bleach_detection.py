# import libraries
import os
import numpy as np
import cv2
import torch

from torchvision import transforms
from torchvision.models import alexnet
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# parameters
dim = 512
patch = 50
cluster_size = 100
k = 4
overlap = 0.5  # overlap ratio
repeats = 2


# dataset directory
bleached_dir = 'C:/Users/cherr/Downloads/archive/Train/Bleached/'
unbleached_dir = "C:/Users/cherr/Downloads/archive/Train/Unbleached/"


# load pretrained AlexNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = alexnet(pretrained=True)
model.eval()  # Set the model to evaluation mode
model.to(device)


# FUNCTIONs
# load images
def load_images(dir):
  images = []
  for filename in os.listdir(dir):
    img = cv2.imread(os.path.join(dir, filename))
    if img is not None:
      img = cv2.resize(img, (dim, dim))
      images.append(img)
  return images

# preprocessing
def preprocess_image(image):
  transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((224, 224)),  # Resize to AlexNet input size
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
  ])
  return transform(image)

# divide the image into patches
def extract_patches(image, patch_size, overlap):
  patches = []
  step = int(patch_size * (1 - overlap))

  rows, cols = image.shape[:2] # get the number of rows and columns in the image
  for y in range(0, rows - patch_size + 1, step):
    for x in range(0, cols - patch_size + 1, step):
      patch = image[y:y + patch_size, x:x + patch_size]
      patches.append(patch)

  return np.array(patches)


# AlexNet
def extract_alexnet_features(patches, model, device):
  features = []

  for patch in patches:
    inputs = preprocess_image(patch).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
    with torch.no_grad():
      alexnet_features = model.features(inputs).flatten().cpu().numpy()  # Extract features and move to CPU
    features.append(alexnet_features)
  return np.array(features)

# ColorTexture
def extract_color_texture_features(patches):
  features = []
  for patch in patches:
    hist_r = cv2.calcHist([patch], [2], None, [256], [0, 256])
    hist_g = cv2.calcHist([patch], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([patch], [0], None, [256], [0, 256])
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
    features.append(hist)
  return np.array(features)

# Hybrid model: AlexNet + ColorTexture
def extract_all_features(images, model, device):
  all_features = []
  for img in images:
    patches = extract_patches(img, patch, overlap)
    alexnet_features = extract_alexnet_features(patches, model, device)
    color_texture_features = extract_color_texture_features(patches)
    
    # Concatenate AlexNet and ColorTexture features
    combined_features = np.hstack([alexnet_features, color_texture_features])
    all_features.append(combined_features)

  return np.array(all_features)


# K-means clustering to create the vocabulary set
def create_vocabulary(raw_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=repeats)
    kmeans.fit(raw_features)
    return kmeans.cluster_centers_, kmeans.labels_

# BoF feature extraction
def bag_of_features_extraction(features, vocab):
    # Assign each feature to the nearest cluster center
    kmeans = KMeans(n_clusters=len(vocab), init=vocab, n_init=1)  # No need to fit again, just predict
    labels = kmeans.predict(features)
    # Generate a histogram of cluster assignments (BoF feature representation)
    hist, _ = np.histogram(labels, bins=len(vocab), range=(0, len(vocab)))
    return hist


# SVM training and classification
def train_svm(train_features, train_labels):
  scaler = StandardScaler()
  train_features_scaled = scaler.fit_transform(train_features)
  svm_classifier = SVC(kernel='poly', degree=2)  # Quadratic kernel
  svm_classifier.fit(train_features_scaled, train_labels)
  return svm_classifier, scaler

def classify_svm(svm_classifier, scaler, test_features):
  test_features_scaled = scaler.transform(test_features)
  return svm_classifier.predict(test_features_scaled)


def stratified_k_fold_validation(bleached_dir, unbleached_dir, num_clusters, model, device, k=4):
  # Load bleached and unbleached images
  bleached_images = load_images(bleached_dir)
  unbleached_images = load_images(unbleached_dir)

  # Assign labels: 1 for bleached, 0 for unbleached
  bleached_labels = np.ones(len(bleached_images))
  unbleached_labels = np.zeros(len(unbleached_images))

  # Combine images and labels
  images = bleached_images + unbleached_images
  labels = np.concatenate([bleached_labels, unbleached_labels])

  # Extract features from images
  raw_features = []
  for img in images:
    patches = extract_patches(img, patch, overlap)
    alexnet_features = extract_alexnet_features(patches, model, device)
    color_texture_features = extract_color_texture_features(patches)
    combined_features = np.hstack([alexnet_features, color_texture_features])
    raw_features.append(combined_features)

  raw_features = np.vstack(raw_features)

  # Perform K-means clustering to create vocabulary
  vocab, _ = create_vocabulary(raw_features, num_clusters)

  # Extract final BoF features for each image
  final_features = []
  for img in images:
    patches = extract_patches(img, patch, overlap)
    alexnet_features = extract_alexnet_features(patches, model, device)
    color_texture_features = extract_color_texture_features(patches)
    combined_features = np.hstack([alexnet_features, color_texture_features])

    # BoF feature extraction
    bof_feature = bag_of_features_extraction(combined_features, vocab)
    final_features.append(bof_feature)

  final_features = np.array(final_features)

  # Stratified K-Fold Cross-Validation
  skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
  accuracies = []

  for fold, (train_idx, test_idx) in enumerate(skf.split(final_features, labels)):
    print(f"Fold {fold + 1}")

    # Split data into training and testing sets for this fold
    X_train, X_test = final_features[train_idx], final_features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Repeat the training and testing process twice
    for repeat in range(repeats):
      print(f"  Repeat {repeat + 1}")

      # Train SVM with the extracted BoF features
      svm_classifier, scaler = train_svm(X_train, y_train)

      # Predict on the test set
      predictions = classify_svm(svm_classifier, scaler, X_test)

      # Calculate accuracy
      accuracy = accuracy_score(y_test, predictions)
      accuracies.append(accuracy)
      print(f"    Accuracy: {accuracy * 100:.2f}%")
      
  # Print average accuracy
  print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")

# Run the StratifiedKFold validation function
stratified_k_fold_validation(bleached_dir, unbleached_dir, cluster_size, model, device, k=4)
