from utils.paddleocr import OCRProcessor


ocr = OCRProcessor('models/ch_ppocrv4_det.onnx',
                   'models/ch_ppocrv4_rec.onnx',
                   'models/rec_word_dict.txt',
                   cls_model_path='models/ch_ppocrv2_cls.onnx',
                   thread_num=3)

results = ocr('images/11.jpg')
print(results)
