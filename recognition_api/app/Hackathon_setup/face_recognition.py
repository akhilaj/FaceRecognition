import os
from PIL import Image
import cv2
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
# Global placeholders to lazily load models only when needed
feature_net = None
clf = None

def load_models_lazy():
    """Lazily load models into RAM only on the first API call to preserve startup memory."""
    global feature_net, clf
    if feature_net is None:
        import torch
        from .face_recognition_model import SiameseNetwork
        
        feature_net = SiameseNetwork()
        ckpt = torch.load(os.path.join(current_path, 'siamese_model.t7'), map_location=torch.device('cpu'))
        feature_net.load_state_dict(ckpt['net_dict'])
        feature_net.eval()
        
    if clf is None:
        import joblib
        classifier_path = os.path.join(current_path, 'decision_tree_model.sav')
        if os.path.exists(classifier_path):
            clf = joblib.load(classifier_path)


def detected_face(image):
    face_haar = os.path.join(current_path, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_haar)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return 0
        
    face_areas = [w * h for (x, y, w, h) in faces]
    max_idx = np.argmax(face_areas)
    x, y, w, h = faces[max_idx]
    return Image.fromarray(gray[y:y+h, x:x+w])


#1) Images captured from mobile is passed as parameter to the below function in the API call. It returns the similarity measure between given images.
#2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
#3) Define an object to your siamese network here in the function and load the weight from the trained network, set it in evaluation mode.
#4) Get the features for both the faces from the network and return the similarity measure, Euclidean,cosine etc can be it. But choose the Relevant measure.
#5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function 
#Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_similarity(img1, img2):
    import torch
    import torch.nn.functional as F
    from .face_recognition_model import trnscm
    
    # Trigger lazy loading safely
    load_models_lazy()
    from PIL import Image
    
    det_img1 = detected_face(img1)
    det_img2 = detected_face(img2)
    
    if det_img1 == 0 or det_img2 == 0:
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
        det_img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))
        
    face1 = trnscm(det_img1).unsqueeze(0)
    face2 = trnscm(det_img2).unsqueeze(0)
    
    with torch.no_grad():
        output1 = feature_net.forward_once(face1)
        output2 = feature_net.forward_once(face2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        
    return float(euclidean_distance.item())
    
    
#1) Image captured from mobile is passed as parameter to this function in the API call, It returns the face class in the string form ex: "Person1"
#2) The image is passed to the function in base64 encoding, Code to decode the image provided within the function
#3) Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode
#4) Perform necessary transformations to the input(detected face using the above function).
#5) Along with the siamese, you need the classifier as well, which is to be finetuned with the faces that you are training
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_face_class(img1):
    import torch
    from .face_recognition_model import trnscm, classes
    from PIL import Image
    
    load_models_lazy()
    if clf is None:
        return "Classifier weights missing"
        
    det_img1 = detected_face(img1)
    if det_img1 == 0:
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))
        
    face1 = trnscm(det_img1).unsqueeze(0)
    
    with torch.no_grad():
        embedding = feature_net.forward_once(face1).numpy()
        
    pred = clf.predict(embedding)[0]
    
    if isinstance(pred, (int, np.integer)):
        return classes[pred] if pred < len(classes) else str(pred)
    return str(pred)
