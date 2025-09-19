import tensorflow as tf

from deepface import DeepFace
import cv2
img_path = "quick_render.png"
result = DeepFace.analyze(img_path, actions=['emotion'])
print(result)