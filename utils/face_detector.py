# 本代码由lasifea编写，参考文档地址：https://github.com/deepinsight/insightface，模型来自insightface
# 适用于insightface开源人脸检测的onnx模型做推理
# 首次完成时间于2024-06-08，最后修改时间于2024-06-13

from onnxruntime import InferenceSession, SessionOptions
import cv2
import numpy as np
from typing import Union, Tuple, List
from .helpers import read_image


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceDetector:
    def __init__(self,
                 model_path: str,
                 max_face_num: int = 100,
                 thread_num: int = 2,
                 use_gpu: bool = False):
        """
        脸部检测器
        :param model_path: onnx模型文件路径
        :param max_face_num: 最多检测多少张人脸
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
        self.max_face_num = max_face_num

        self._box_side = 640
        self._nms_thresh = 0.4
        self._input_mean = 127.5
        self._input_std = 128.0
        self._use_kps = True
        self._num_anchors = 2
        self._fmc = 3
        self._feat_stride_fpn = (8, 16, 32)

        self._author = 'lasifea'
        self._email = 'lasifea@163.com'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _preprocess(self, img_obj: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        将图像等比例缩放至最大边长为_box_side，再填充
        :param img_obj: 图片对象
        :return: 返回两个参数，第一个参数为处理后的图像，第二个参数为图片的缩放倍数
        """
        img = read_image(img_obj)
        h, w = img.shape[:2]

        scale = self._box_side / max(h, w)
        obj_size = (self._box_side, round(h * scale)) if w > h else (round(w * scale), self._box_side)
        img2 = cv2.resize(img, obj_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        # img2 = cv2.resize(img, obj_size)
        h, w = img2.shape[:2]

        mask = np.zeros((self._box_side, self._box_side, 3), dtype=np.uint8)
        mask[:h, :w] = img2
        return mask, scale

    def _preprocess2(self, img: np.ndarray) -> np.ndarray:
        """
        图像预处理第二步
        :param img: 预处理后的图像
        :return: 可供模型输入的数据
        """
        input_tensor = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        input_tensor -= self._input_mean
        input_tensor /= self._input_std
        return input_tensor[np.newaxis, ...]

    def _postprocess(self, outputs_tensor: np.ndarray, scale: float, conf: float) -> List[dict]:
        """
        后处理，从模型推理结果中提取有价值的数据
        :param outputs_tensor: 模型推理结果
        :param scale: 原图的缩放比例
        :param conf: 人脸检测的最低置信度
        :return: 返回一个包含置信度、脸部xyxy位置，脸部关键点位置的字典组成的列表
        """
        scores_li, boxes_li, points_li = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs_tensor[idx]
            boxes = outputs_tensor[idx + self._fmc]
            boxes *= stride

            height = self._box_side // stride
            width = self._box_side // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32).reshape((-1, 2))
            anchor_centers *= stride
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            valid_idx = np.where(scores >= conf)[0]
            boxes = distance2bbox(anchor_centers, boxes)
            scores_li.append(scores[valid_idx])
            boxes_li.append(boxes[valid_idx])

            if self._use_kps:
                points = outputs_tensor[idx + self._fmc * 2]
                points *= stride
                points = distance2kps(anchor_centers, points)
                points_li.append(points[valid_idx])

        scores = np.vstack(scores_li).ravel()
        order = scores.argsort()[::-1]
        boxes = np.vstack(boxes_li)
        boxes /= scale
        boxes = boxes[order]

        keep = self._nms(boxes)
        scores = scores[order][keep]
        boxes = boxes[keep]

        if self._use_kps:
            points = np.vstack(points_li)
            points /= scale
            points = points[order][keep]

        results = []
        for idx, i in enumerate(scores[:self.max_face_num]):
            obj_points = points[idx].reshape(-1, 2) if self._use_kps else []
            dic = {'score': i, 'box': boxes[idx], 'points': obj_points}
            results.append(dic)

        return results

    def _nms(self, dets: np.ndarray) -> list:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.array(range(dets.shape[0]))

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self._nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def forward(self, img_obj: Union[str, bytes, np.ndarray], conf: float = 0.5) -> List[dict]:
        """
        输入图像得到人脸相关数据
        :param img_obj: 图片对象
        :param conf: 人脸检测的最低置信度
        :return: 返回一个包含置信度、脸部xyxy位置，脸部关键点位置的字典组成的列表
        """
        input_img, scale = self._preprocess(img_obj)
        input_tensor = self._preprocess2(input_img)

        outputs_tensor = self.model.run(None, {self.input_name: input_tensor})
        return self._postprocess(outputs_tensor, scale, conf)
