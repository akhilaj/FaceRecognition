import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
#print(sys.path)
from typing import Any

from fastapi import FastAPI, Request, APIRouter, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app import __version__
from app.Hackathon_setup import face_recognition

import numpy as np
from PIL import Image


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# To store files uploaded by users
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# To access Templates directory
templates = Jinja2Templates(directory="app/templates")

simi_filename1 = None
simi_filename2 = None
face_rec_filename = None
expr_rec_filename = None


####################################  Home Page endpoints  #################################################
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


####################################  Face Similarity endpoints  #################################################
@app.get("/similarity/")
async def similarity_root(request: Request):
    return templates.TemplateResponse(request, "similarity.html")


@app.post("/predict_similarity/")
async def predict_similarity(request: Request, file1: UploadFile = File(...), file2: UploadFile = File(...)):
    simi_filename1 = 'app/static/' + file1.filename
    simi_filename2 = 'app/static/' + file2.filename
    
    with open(simi_filename1, "wb") as f:
        f.write(await file1.read())
    with open(simi_filename2, "wb") as f:
        f.write(await file2.read())

    img1 = Image.open(simi_filename1).convert("RGB")
    img1 = np.array(img1).astype(np.uint8)
    
    img2 = Image.open(simi_filename2).convert("RGB")
    img2 = np.array(img2).astype(np.uint8)

    # Calculate similarity score through Pillow-based framework
    score = face_recognition.get_similarity(img1, img2)
    result_str = f"{score:.4f}"

    # os.remove(simi_filename1)
    # os.remove(simi_filename2)

    # FIXED: Explicitly pass request first, and use strict string mapping arrays
    return templates.TemplateResponse(
        request, 
        "predict_similarity.html", 
        {
            "result": result_str, 
            "simi_filename1": simi_filename1, 
            "simi_filename2": simi_filename2
        }
    )
    

####################################  Face Recognition endpoints  #################################################
@app.get("/face_recognition/")
async def face_recognition_root(request: Request):
    return templates.TemplateResponse(request, "face_recognition.html")


@app.post("/predict_face_recognition/")
async def predict_face_recognition(request: Request, file1: UploadFile = File(...)):
    face_rec_filename = 'app/static/' + file1.filename
    with open(face_rec_filename, "wb") as f:
        f.write(await file1.read())

    img1 = Image.open(face_rec_filename).convert("RGB")
    img1 = np.array(img1).astype(np.uint8)

    # Run inference pipeline using our lazy loaded weights setup
    predicted_class = face_recognition.get_face_class(img1)
    result_str = str(predicted_class)

    # os.remove(face_rec_filename)

    # FIXED: Explicitly pass request first, and use strict string mapping arrays
    return templates.TemplateResponse(
        request, 
        "predict_face_recognition.html", 
        {
            "result": result_str, 
            "face_rec_filename": face_rec_filename
        }
    )

####################################  Expresion Recognition endpoints  #################################################
# Placeholders to prevent HTML template matching errors for the bypassed module
@app.get("/expression/", name="expr_recognition_root")
async def expression_root(request: Request):
    return templates.TemplateResponse(
        request, 
        "expression.html", 
        {"result": "Expression Recognition module is currently disabled to optimize system resources."}
    )

@app.post("/predict_expression/", name="predict_expression")
async def predict_expression_placeholder(request: Request):
    return templates.TemplateResponse(
        request, 
        "expression_result.html", 
        {"result": "Expression Recognition module is currently disabled to optimize system resources."}
    )

# @app.get("/expr_recognition/")
# async def expr_recognition_root(request: Request):
#     return templates.TemplateResponse("expr_recognition.html", {'request': request,})


# @app.post("/predict_expr_recognition/")
# async def create_upload_files(request: Request, file4: UploadFile = File(...)):
#     global expr_rec_filename
    
#     if 'image' in file4.content_type:
#         contents = await file4.read()
#         expr_rec_filename = 'app/static/' + file4.filename
#         with open(expr_rec_filename, 'wb') as f:
#             f.write(contents)
    
#     img1 = Image.open(expr_rec_filename)
#     img1 =  np.array(img1).reshape(img1.size[1], img1.size[0], 3).astype(np.uint8)
        
#     result = exp_recognition.get_expression(img1)
#     print(result)
    
#     return templates.TemplateResponse("predict_expr_recognition.html", {"request": request,
#                                                        "result": result,
#                                                        "expr_rec_filename": '../static/'+file4.filename,})



# # Set all CORS enabled origins
# if settings.BACKEND_CORS_ORIGINS:
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
#         allow_credentials=True,
#         allow_methods=["*"],
#         allow_headers=["*"],
#     )


# Start app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
