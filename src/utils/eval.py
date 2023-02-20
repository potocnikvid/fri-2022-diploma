import numpy as np
import wandb
import sys

from utils.core import assert_batch_shape
from utils.viz import image_add_label

from pprint import pprint

from deepface import DeepFace

class BaseEvaluator():
    def __init__(self):
        self.deepface = DeepFace
        self.eval_models = [
                "VGG-Face", 
                "Facenet", 
                "Facenet512", 
                "OpenFace", 
                "DeepFace", 
                "DeepID", 
                "ArcFace", 
                "Dlib", 
                "SFace",
            ]
        
        
    def cos_sim(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def compare_embeddings(self, first_e, second_e):
        return self.cos_sim(first_e, second_e)

    def get_embedding(self, image, model, model_name):
        return np.array(self.deepface.represent(image, model=model, model_name=model_name, enforce_detection=False))

    def get_embedding_all_models(self, image):
        embeddings = {}
        for model_name in self.eval_models:
            embeddings[model_name] = self.get_embedding(image, model=None, model_name=model_name)
        return embeddings
    
    def get_embeddings(self, images, model, model_name):
        embeddings = []
        for image in images:
            embeddings.append(self.get_embedding(image, model, model_name))
        return np.array(embeddings)


    def evaluate(self, image, image_hat, model_name):
        model = DeepFace.build_model(model_name)
        assert_batch_shape(image, (18, 512))
        assert_batch_shape(image_hat, (18, 512))
        image_embedding = self.get_embedding(image, model, model_name)
        image_hat_embedding = self.get_embedding(image_hat, model, model_name)
        res = self.compare_embeddings(image_embedding[0], image_hat_embedding[0])
        return res

    def evaluate_batch(self, images, images_hat, model_name):
        model = DeepFace.build_model(model_name)
        print(model)
        image_embeddings = self.get_embeddings(images, model, model_name)
        image_hat_embeddings = self.get_embeddings(images_hat, model, model_name)
        res = []
        for i in range(len(image_embeddings)):
            res.append(self.compare_embeddings(image_embeddings[i], image_hat_embeddings[i]))
        return res

