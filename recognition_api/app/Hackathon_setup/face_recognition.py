import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
# In the below line,remove '.' while working on your local system. However Make sure that '.' is present before face_recognition_model while uploading to the server, Do not remove it.
from .face_recognition_model import *
from PIL import Image
import base64
import io
import os
import joblib
import pickle
# Add more imports if required



###########################################################################################################################################
#         Caution: Don't change any of the filenames, function names and definitions                                                      #
#        Always use the current_path + file_name for refering any files, without it we cannot access files on the server                  # 
###########################################################################################################################################

# Current_path stores absolute path of the file from where it runs. 
current_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")
#1) The below function is used to detect faces in the given image.
#2) It returns only one image which has maximum area out of all the detected faces in the photo.
#3) If no face is detected,then it returns zero(0).
# 1. Load Siamese Network once at startup
feature_net = SiameseNetwork().to(device)
ckpt = torch.load(os.path.join(current_path, 'siamese_model.t7'), map_location=device)
feature_net.load_state_dict(ckpt['net_dict'])
feature_net.eval()

# 2. Load Sklearn Classifier once at startup (update filename if yours differs)
classifier_path = os.path.join(current_path, 'decision_tree_model.sav')
clf = joblib.load(classifier_path) if os.path.exists(classifier_path) else None

def detected_face(image):
    eye_haar = current_path + '/haarcascade_eye.xml'
    face_haar = current_path + '/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_haar)
    eye_cascade = cv2.CascadeClassifier(eye_haar)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas=[]
    images = []
    required_image=0
    for i, (x,y,w,h) in enumerate(faces):
        face_cropped = gray[y:y+h, x:x+w]
        face_areas.append(w*h)
        images.append(face_cropped)
        required_image = images[np.argmax(face_areas)]
        required_image = Image.fromarray(required_image)
    return required_image


#1) Images captured from mobile is passed as parameter to the below function in the API call. It returns the similarity measure between given images.
#2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
#3) Define an object to your siamese network here in the function and load the weight from the trained network, set it in evaluation mode.
#4) Get the features for both the faces from the network and return the similarity measure, Euclidean,cosine etc can be it. But choose the Relevant measure.
#5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function 
#Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_similarity(img1, img2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    det_img1 = detected_face(img1)
    det_img2 = detected_face(img2)
    if(det_img1 == 0 or det_img2 == 0):
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        det_img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    face1 = trnscm(det_img1).unsqueeze(0)
    face2 = trnscm(det_img2).unsqueeze(0)
    ##########################################################################################
    ##Example for loading a model using weight state dictionary:                            ##
    ## feature_net = light_cnn() #Example Network                                           ##
    ## model = torch.load(current_path + '/siamese_model.t7', map_location=device)          ##
    ## feature_net.load_state_dict(model['net_dict'])                                       ##
    ##                                                                                      ##
    ##current_path + '/<network_definition>' is path of the saved model if present in       ##
    ##the same path as this file, we recommend to put in the same directory                 ##
    ##########################################################################################
    ##########################################################################################
    
    # YOUR CODE HERE, load the model
    
    # YOUR CODE HERE, return similarity measure using your model
    with torch.no_grad():
            output1, output2 = feature_net(face1, face2)
            euclidean_distance = F.pairwise_distance(output1, output2)
            
    return float(euclidean_distance.item())
    
#1) Image captured from mobile is passed as parameter to this function in the API call, It returns the face class in the string form ex: "Person1"
#2) The image is passed to the function in base64 encoding, Code to decode the image provided within the function
#3) Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode
#4) Perform necessary transformations to the input(detected face using the above function).
#5) Along with the siamese, you need the classifier as well, which is to be finetuned with the faces that you are training
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_face_class(img1):
    if clf is None:
        return "Classifier model file not found"
        
    det_img1 = detected_face(img1)
    if det_img1 == 0:
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
        
    face1 = trnscm(det_img1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = feature_net.forward_once(face1).cpu().numpy()
        
    pred = clf.predict(embedding)[0]
    
    # If classifier outputs an integer index (e.g., 0, 1, 2), map it using classes
    if isinstance(pred, (int, np.integer)):
        return classes[pred] if pred < len(classes) else str(pred)
    return str(pred)
