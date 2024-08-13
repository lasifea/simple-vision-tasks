# 本代码由lasifea编写，参考文档地址：https://github.com/PaddlePaddle/PaddleOCR，https://github.com/hpc203/PaddleOCR-v3-onnxrun-cpp-py
# 适用于文字识别
# 首次完成时间于2024-01-28，最后修改时间于2024-08-13

from onnxruntime import SessionOptions, InferenceSession
import cv2
import numpy as np
from pyclipper import PyclipperOffset, JT_ROUND, ET_CLOSEDPOLYGON
from math import ceil
from typing import Union, Tuple, Iterator, Optional, List
from .helpers import read_image


class TextDetector:
    def __init__(self, model_path: str,
                 min_score: float = 0.6,
                 thread_num: int = 2,
                 use_gpu: bool = False):
        """
        文字检测器
        :param model_path: 文字检测模型的路径
        :param min_score: 满足文本框检测的最低置信度
        :param thread_num: 线程数量，默认2个线程
        :param use_gpu: 是否使用显卡推理，目前仅支持cuda
        """
        if use_gpu:
            import os
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
            providers = ('CUDAExecutionProvider', 'CPUExecutionProvider')
        else:
            providers = ('CPUExecutionProvider',)

        options = SessionOptions()
        options.log_severity_level = 3
        if thread_num > 0:
            options.intra_op_num_threads = thread_num

        self.model = InferenceSession(model_path, providers=providers, sess_options=options)
        self.input_name = self.model.get_inputs()[0].name
        self._limit_side_len = 960
        self._input_mean = np.float32([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self._input_std = np.float32([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        self.min_size = 4
        self.threshold = 0.3
        self.box_threshold = min_score
        self.max_candidates = 1000
        self.expansion_ratio = 1.6

        self._author = 'lasifea'
        self._email = 'lasifea@163.com'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _preprocess(self, img_obj: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        图像预处理，限制输入图像尺寸
        :param img_obj: 图片对象
        :return: 返回四个参数，第一个为处理后的图片，第二个参数为原始图像，第三四个参数分别为图像的宽高
        """
        img = read_image(img_obj)
        h, w = img.shape[:2]
        if (max_side := max(h, w)) > self._limit_side_len:
            obj_w, obj_h = (self._limit_side_len, self._limit_side_len / w * h) if w > h else (self._limit_side_len / h * w, self._limit_side_len)
        else:
            obj_w, obj_h = w, h

        obj_w = max(int(round(obj_w / 32) * 32), 32)
        obj_h = max(int(round(obj_h / 32) * 32), 32)

        img2 = cv2.resize(img, (obj_w, obj_h), interpolation=cv2.INTER_AREA if max_side > self._limit_side_len else cv2.INTER_LINEAR)
        return img2, img, w, h

    def _preprocess2(self, input_img: np.ndarray) -> np.ndarray:
        """
        图像预处理第二步
        :param input_img: 预处理后的图像
        :return: 可供模型输入的数据
        """
        input_tensor = input_img.astype(np.float32)
        input_tensor /= 255
        input_tensor -= self._input_mean
        input_tensor /= self._input_std
        return input_tensor.transpose((2, 0, 1))[np.newaxis, ...]

    def _postprocess(self, confidence_mask: np.ndarray, ori_w: int, ori_h: int) -> List[dict]:
        """
        后处理，得到文本区域信息
        :param confidence_mask: 模型推理结果
        :param ori_w: 原图的宽
        :param ori_h: 原图的高
        :return: 返回包含文字区域位置、文本区域置信度、文本区域中心点的字典组成的列表
        """
        height, width = confidence_mask.shape
        mask = (confidence_mask > self.threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ocr_results = []

        for contour in contours[:self.max_candidates]:
            rect_obj = cv2.minAreaRect(contour)
            if min(rect_obj[1]) < self.min_size:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            mask2 = np.zeros((h, w), dtype=np.uint8)
            contour2 = contour.copy()
            contour2[..., 0] = contour2[..., 0] - x
            contour2[..., 1] = contour2[..., 1] - y
            cv2.fillPoly(mask2, [contour2], (1,))
            score = confidence_mask[y:y+h, x:x+w][mask2 == 1].mean()

            if score < self.box_threshold:
                continue

            points = cv2.boxPoints(rect_obj)
            distance = cv2.contourArea(points) * self.expansion_ratio / cv2.arcLength(points, True)
            offset = PyclipperOffset()
            offset.AddPath(points, JT_ROUND, ET_CLOSEDPOLYGON)
            new_contour = np.array(offset.Execute(distance)[0])
            rect_obj = cv2.minAreaRect(new_contour)
            if min(rect_obj[1]) < self.min_size + 2:
                continue

            points = cv2.boxPoints(rect_obj)
            points[:, 0] = points[:, 0] / width * ori_w
            points[:, 1] = points[:, 1] / height * ori_h
            center_point = (rect_obj[0][0] / width * ori_w, rect_obj[0][1] / height * ori_h)
            dic = {'points': points, 'score': score, 'center_point': center_point}
            ocr_results.append(dic)

        return ocr_results

    def forward(self, img_obj: Union[str, bytes, np.ndarray]) -> Tuple[list, np.ndarray]:
        """
        输入图像得到文本框信息
        :param img_obj: 图片对象
        :return: 返回两个参数，第一个为包含文字区域位置、文本区域置信度、文本区域中心点的字典组成的列表，第二个参数为BGR格式的ndarray原始图像
        """
        input_img, ori_img, ori_w, ori_h = self._preprocess(img_obj)
        input_tensor = self._preprocess2(input_img)
        output_tensor = self.model.run(None, {self.input_name: input_tensor})[0][0, 0]
        return self._postprocess(output_tensor, ori_w, ori_h), ori_img

    @staticmethod
    def warp_box(det_results: List[dict], ori_img: np.ndarray) -> Iterator:
        """
        将原图中不同角度的文本框做放射变换，用于后续文本方向判断及文字识别
        :param det_results: 文字检测器返回的结果
        :param ori_img: 原始图像
        :return: 返回包含角度纠正的的文本框图片的生成器
        """
        for result in det_results:
            start_idx = result['points'].sum(axis=1).argmin()
            points = np.roll(result['points'], 4 - start_idx, axis=0)
            w = int(np.linalg.norm(points[0] - points[1])) + 1
            h = int(np.linalg.norm(points[0] - points[3])) + 1
            new_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

            m = cv2.getPerspectiveTransform(points, new_points)
            warp_img = cv2.warpPerspective(ori_img, m, (w, h))
            if h / w >= 1.5:
                warp_img = np.rot90(warp_img)
            yield warp_img


class TextClassifier:
    def __init__(self, model_path: str,
                 cls_threshold: float = 0.9,
                 thread_num: int = 2,
                 use_gpu: bool = False):
        """
        文本方向分类器
        :param model_path: 文字方向分类器模型的路径
        :param cls_threshold: 满足方向分类的最低置信度
        :param thread_num: 线程数量，默认2个线程
        :param use_gpu: 是否使用显卡推理，目前仅支持cuda
        """
        if use_gpu:
            import os
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
            providers = ('CUDAExecutionProvider', 'CPUExecutionProvider')
        else:
            providers = ('CPUExecutionProvider',)

        options = SessionOptions()
        options.log_severity_level = 3
        if thread_num > 0:
            options.intra_op_num_threads = thread_num

        self.model = InferenceSession(model_path, providers=providers, sess_options=options)
        self.cls_threshold = cls_threshold
        self.input_name = self.model.get_inputs()[0].name
        self._input_size = (3, 48, 192)
        self._labels = (0, 180)
        self._input_mean = 127.5
        self._input_std = 127.5

        self._author = 'lasifea'
        self._email = 'lasifea@163.com'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _preprocess(self, img_obj: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        图像预处理，将文本框图像的高等比例固定尺寸
        :param img_obj: 图片对象
        :return: 返回处理后的图像
        """
        img = read_image(img_obj)
        h, w = img.shape[:2]
        scale = self._input_size[1] / h
        obj_w = ceil(w * scale)
        # if obj_w > self._input_size[2]:
        #     obj_w = self._input_size[2]

        img2 = cv2.resize(img, (obj_w, self._input_size[1]), interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_CUBIC)
        return img2

    def _preprocess2(self, input_img: np.ndarray) -> np.ndarray:
        """
        图像预处理第二步
        :param input_img: 预处理后的图像
        :return: 可供模型输入的数据
        """
        input_tensor = input_img.transpose((2, 0, 1)).astype(np.float32)
        input_tensor -= self._input_mean
        input_tensor /= self._input_std
        return input_tensor[np.newaxis, ...]

    def _postprocess(self, each_output: np.ndarray) -> int:
        """
        后处理，判断图像角度
        :param each_output: 模型推理结果
        :return: 返回图像角度，0或180
        """
        idx = each_output.argmax()
        score = np.max(each_output)
        return self._labels[1] if idx == 1 and score >= self.cls_threshold else self._labels[0]

    def forward(self, img_obj: Union[str, bytes, np.ndarray]) -> int:
        """
        输入图像得到角度
        :param img_obj: 图片对象
        :return: 返回图像角度，0或180
        """
        input_img = self._preprocess(img_obj)
        input_tensor = self._preprocess2(input_img)
        outputs = self.model.run(None, {self.input_name: input_tensor})[0][0]
        return self._postprocess(outputs)


class TextRecognizer:
    def __init__(self, model_path: str,
                 text_path: str,
                 rec_threshold: float = 0.5,
                 thread_num: int = 2,
                 use_gpu: bool = False):
        """
        文本识别器
        :param model_path: 文字识别模型的路径
        :param text_path: 文本库的路径
        :param rec_threshold: 文字识别的置信度，存在意义不大
        :param thread_num: 线程数量，默认2个线程
        :param use_gpu: 是否使用显卡推理，目前仅支持cuda
        """
        if use_gpu:
            import os
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
            providers = ('CUDAExecutionProvider', 'CPUExecutionProvider')
        else:
            providers = ('CPUExecutionProvider',)

        options = SessionOptions()
        options.log_severity_level = 3
        if thread_num > 0:
            options.intra_op_num_threads = thread_num

        self.model = InferenceSession(model_path, providers=providers, sess_options=options)
        self.rec_threshold = rec_threshold
        self.input_name = self.model.get_inputs()[0].name
        self._input_size = (3, 48, 320)
        self._input_mean = 127.5
        self._input_std = 127.5
        with open(text_path, 'r', encoding='utf8') as f:
            self._texts = f.read().replace('\n', '') + ' '

        self._author = 'lasifea'
        self._email = 'lasifea@163.com'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _preprocess(self, img_obj: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """
        图像预处理，将文本框图像的高等比例固定尺寸
        :param img_obj: 图片对象
        :return: 返回处理后的图像
        """
        img = read_image(img_obj)
        h, w = img.shape[:2]
        scale = self._input_size[1] / h
        obj_w = ceil(w * scale)

        img2 = cv2.resize(img, (obj_w, self._input_size[1]), interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_CUBIC)
        return img2

    def _preprocess2(self, input_img: np.ndarray) -> np.ndarray:
        """
        图像预处理第二步
        :param input_img: 预处理后的图像
        :return: 可供模型输入的数据
        """
        input_tensor = input_img.transpose((2, 0, 1)).astype(np.float32)
        input_tensor -= self._input_mean
        input_tensor /= self._input_std
        return input_tensor[np.newaxis, ...]

    def _postprocess(self, each_output: np.ndarray) -> str:
        """
        后处理，判断文字识别结果
        :param each_output: 模型推理结果
        :return: 文字识别结果
        """
        text_idx_li = each_output.argmax(axis=1)
        content = ''.join([self._texts[i - 1] for idx, i in enumerate(text_idx_li) if i != 0 and not (idx > 0 and text_idx_li[idx - 1] == text_idx_li[idx])])
        return content

    def forward(self, img_obj: Union[str, bytes, np.ndarray]) -> str:
        """
        输入图像得到文字识别结果
        :param img_obj: 图片对象
        :return: 文字识别结果
        """
        input_img = self._preprocess(img_obj)
        input_tensor = self._preprocess2(input_img)
        outputs = self.model.run(None, {self.input_name: input_tensor})[0][0]
        return self._postprocess(outputs)


class OCRProcessor:
    def __init__(self, det_model_path: str,
                 rec_model_path: str,
                 text_path: str,
                 cls_model_path: Optional[str] = None,
                 thread_num: int = 2,
                 use_gpu: bool = False,
                 save_warp_img: bool = False):
        """
        文字识别
        :param det_model_path: 文字检测模型的路径
        :param rec_model_path: 文字识别模型的路径
        :param text_path: 文本库的路径
        :param cls_model_path: 文字方向分类器模型的路径，如不需要方向检测可取消此项以提升速度
        :param thread_num: 模型检测使用的线程数，默认使用两个线程
        :param use_gpu: 模型检测使用的线程数，默认使用两个线程
        :param save_warp_img: 保存每个文本区域图片，默认不保存
        """
        self.text_detector = TextDetector(det_model_path, thread_num=thread_num, use_gpu=use_gpu)
        self.text_recognizer = TextRecognizer(rec_model_path, text_path, thread_num=thread_num, use_gpu=use_gpu)
        self.text_classifier = TextClassifier(cls_model_path, thread_num=thread_num, use_gpu=use_gpu) if cls_model_path else None
        self.save_warp_img = save_warp_img

        self._author = 'lasifea'
        self._email = 'lasifea@163.com'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img_obj: Union[str, bytes, np.ndarray]) -> List[dict]:
        """
        输入图像得到文字识别后的结果
        :param img_obj: 图像路径，图像字节数据或BGR格式的ndarray数组
        :return: 返回包含识别结果、文字区域位置、文本区域置信度、文本区域中心点的字典组成的列表
        """
        results, ori_img = self.text_detector.forward(img_obj)
        ocr_results = []

        for idx, i in enumerate(self.text_detector.warp_box(results, ori_img)):
            if self.text_classifier is not None:
                angle = self.text_classifier.forward(i)
                if angle == 180:
                    i = cv2.rotate(i, cv2.ROTATE_180)

            content = self.text_recognizer.forward(i)
            if not content:
                continue

            results[idx]['content'] = content
            ocr_results.append(results[idx])
            if self.save_warp_img:
                cv2.imwrite(f'{idx}.jpg', i)

        return ocr_results[::-1]


def main():
    from argparse import ArgumentParser
    import time

    parse = ArgumentParser(description='文字识别')
    parse.add_argument('-i', '--input_image', required=True, metavar='', help='需要检测的图像')
    parse.add_argument('-m1', '--det_model_path', required=True, metavar='', help='文字检测模型的路径')
    parse.add_argument('-m2', '--rec_model_path', required=True, metavar='', help='文字识别模型的路径')
    parse.add_argument('-t', '--text_path', required=True, metavar='', help='文本库的路径')
    parse.add_argument('-m3', '--cls_model_path', default='', metavar='', help='文字方向分类器模型的路径，如不需要方向检测可取消此项以提升速度')
    parse.add_argument('-n', '--thread_num', type=int, default=2, metavar='', help='模型检测使用的线程数，默认使用两个线程')
    parse.add_argument('--use_gpu', action='store_true', default=False, help='是否启用GPU检测，默认使用CPU')
    parse.add_argument('--save_warp_img', action='store_true', default=False, help='保存每个文本区域图片，默认不保存')
    args = parse.parse_args()

    ocr = OCRProcessor(args.det_model_path, args.rec_model_path, args.text_path, args.cls_model_path, args.thread_num,
                       args.use_gpu, args.save_warp_img)
    start = time.time()
    results = ocr(args.input_image)
    print(f'图片检测总用时：{time.time()-start} s')
    print(results)


if __name__ == '__main__':
    main()
