# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 21:15:20 2026

@author: Antonin
"""

#IA de détection chat et chien avec scykitlearn
#25/02/2026

import kagglehub
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog, local_binary_pattern
from sklearn import metrics

# Download latest version
path = kagglehub.dataset_download("tongpython/cat-and-dog")

print("Path to dataset files:", path)
chemin = f"{path}\\training_set\\training_set"

#Extract HOG features (textures and shapes of the picture gray_img)
def extract_hog(gray_img):
    feats = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector = True
        )
    return feats

#Extract LBP features (local texture patterns of the picture gray_img)
def extract_lbp(gray_img):
    lbp = local_binary_pattern(gray_img, P = 8, R = 1, method = "uniform")
    
    hist, _ = np.histogram(lbp.ravel(), bins = np.arange(0, 10), range = (0, 9))
    
    hist = hist.astype("float32")
    hist /= (hist.sum()+1e-8)
    
    return hist

#Converts a picture into features
#Argument : path to the picture
#Value : list of the features HOG and LBP
def image_to_features(img_path):
    img = cv2.imread(img_path)
    if img is None : 
        raise ValueError(f"Image illisible/corrompue: {img_path}")
    
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    features.append(extract_hog(gray))
    features.append(extract_lbp(gray))
    return np.concatenate(features)

class_names = [name for name in os.listdir(chemin) if os.path.isdir(os.path.join(chemin, name))]
class_names.sort()


###Main program###
def main():
    X_train = []
    y_train = []
    
    #Extraction and conversion of the pictures of train
    for idc, nameClass in enumerate(class_names) :
        cheminClass = os.path.join(chemin, nameClass)
        for fichier in os.listdir(cheminClass):
            if fichier.lower().endswith(".jpg"):
                cheminFichier = os.path.join(cheminClass, fichier)
                
                vecteur = image_to_features(cheminFichier)
                X_train.append(vecteur)
                y_train.append(idc)
                
    X = np.vstack(X_train)
    y = np.array(y_train, dtype=np.int64)
                
    print("Classes:", class_names)
    print("X shape:", X.shape, "| y shape:", y.shape)
                
    #Extraction et conversion des images de test
    cheminTest = f"{path}\\test_set\\test_set"
    class_names2 = [name for name in os.listdir(cheminTest) if os.path.isdir(os.path.join(cheminTest, name))]
    class_names2.sort()
                
    X_test = []
    y_test = []
    #Extraction and conversion of the pictures of test
    for idc, nameClass in enumerate(class_names2):
        cheminClass = os.path.join(cheminTest, nameClass)
        for fichier in os.listdir(cheminClass):
            if fichier.lower().endswith(".jpg"):
                cheminFichier = os.path.join(cheminClass, fichier)
                vecteur = image_to_features(cheminFichier)
                
                X_test.append(vecteur)
                y_test.append(idc)
    
    #Training of the randomforest
    rF = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=-1)
    rF.fit(X, y)

    #Prediction on test pictures
    y_pred = rF.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    
main()