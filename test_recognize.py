import cv2
from utils.face_detector import FaceDetector
from utils.face_recognizer import FaceRecognizer, one2one, one2many


img = cv2.imread('images/1.jpg')
img2 = cv2.imread('images/2.jpg')
detector = FaceDetector('models/det_500m.onnx')
recognizer = FaceRecognizer('models/w600k_mbf.onnx')

results = detector.forward(img)
results2 = detector(img2)
known_embedding = recognizer.forward(img, results[0]['points'])
unknown_embedding = recognizer(img2, results2[0]['points'])
similarity = one2one(known_embedding, unknown_embedding)

print('人脸相似度：', similarity)
