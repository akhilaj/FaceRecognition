import os
import gc  
import numpy as np
from PIL import Image, ImageOps

current_path = os.path.dirname(os.path.abspath(__file__))

# Global placeholders to keep memory empty at boot
feature_net = None
clf = None
device = None

def load_models_lazy():
    """Imports PyTorch and loads weights into RAM only when an image upload occurs."""
    global feature_net, clf, device
    if feature_net is None:
        import torch
        from .face_recognition_model import SiameseNetwork
        
        device = torch.device("cpu")  # Force strict CPU processing for 4GB RAM stability
        feature_net = SiameseNetwork().to(device)
        
        # Load Siamese weights safely
        model_path = os.path.join(current_path, 'siamese_model.t7')
        ckpt = torch.load(model_path, map_location=device)
        feature_net.load_state_dict(ckpt['net_dict'])
        feature_net.eval()
        
    if clf is None:
        import joblib
        classifier_path = os.path.join(current_path, 'decision_tree_model.sav')
        if os.path.exists(classifier_path):
            clf = joblib.load(classifier_path)

def detected_face_pil(pil_image):
    """
    Pure Pillow replacement for face processing.
    Converts the image to grayscale and safely passes it to the Siamese network.
    """
    # Convert image to grayscale ('L' mode) cleanly using Pillow
    return ImageOps.grayscale(pil_image)

def get_similarity(img1, img2):
    import torch
    import torch.nn.functional as F
    from .face_recognition_model import trnscm
    
    load_models_lazy()
    
    # Convert input numpy arrays directly into pure Pillow Images
    pil_img1 = Image.fromarray(img1)
    pil_img2 = Image.fromarray(img2)
    
    det_img1 = detected_face_pil(pil_img1)
    det_img2 = detected_face_pil(pil_img2)
        
    face1 = trnscm(det_img1).unsqueeze(0).to(device)
    face2 = trnscm(det_img2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output1, output2 = feature_net(face1, face2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        res = float(euclidean_distance.item())
        
    # Clean intermediate tensors immediately to prevent freezing Windows
    del face1, face2, output1, output2
    gc.collect()
    return res

def get_face_class(img1):
    import torch
    from .face_recognition_model import trnscm, classes
    
    load_models_lazy()
    if clf is None:
        return "Classifier weights missing"
        
    pil_img1 = Image.fromarray(img1)
    det_img1 = detected_face_pil(pil_img1)
        
    face1 = trnscm(det_img1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = feature_net.forward_once(face1).cpu().numpy()
        
    pred = clf.predict(embedding)
    
    # Clean workspace extraction vectors
    del face1, embedding
    gc.collect()
    
    if isinstance(pred, (int, np.integer)):
        return classes[pred] if pred < len(classes) else str(pred)
    return str(pred)