import cv2
from utils.face_detector import FaceDetector
from utils.face_landmarker import Face2DLandmarker
from utils.face_recognizer import FaceRecognizer, one2one, one2many
from time import time

img = cv2.imread('images/3.jpg')
detector = FaceDetector('models/det_500m.onnx')
landmarker = Face2DLandmarker('models/2d106det.onnx')
recognizer = FaceRecognizer('models/w600k_mbf.onnx')

start = time()
results = detector(img)
print(f'detector spent -----> {time()-start} s')

start = time()
landmarks = landmarker.forward(img, results[0]['box'])
print(f'landmarker spent -----> {time()-start} s')

start = time()
embedding = recognizer.forward(img, results[0]['points'])
print(f'recognizer spent -----> {time()-start} s')

print(results)
# print(landmarks)
# print(embedding)

# box = results[0]['box'].tolist()
# cv2.rectangle(img, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (0, 255, 255), 1, 16)
# for i in landmarks.tolist():
#     cv2.circle(img, (round(i[0]), round(i[1])), 1, (85, 85, 255), -1, 16)
#
# cv2.imshow('win', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
