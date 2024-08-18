# 本代码由lasifea编写，参考文档地址：https://github.com/deepinsight/insightface，模型来自insightface
# 适用于insightface开源人脸识别的onnx模型做推理
# 首次完成时间于2024-06-13，最后修改时间于2024-08-18

import cv2
import numpy as np
from typing import Union, Tuple
from .helpers import read_image, BaseVisionTask


def one2one(embedding: np.ndarray, embedding2: np.ndarray) -> float:
    """
    两组特征对比
    :param embedding: 一组特征
    :param embedding2: 另一组特征
    :return: 特征相似度
    """
    sim = np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))
    return float(sim)


def one2many(unknown_embedding: np.ndarray, know_embeddings: np.ndarray) -> Tuple[int, float]:
    """
    一组未知特征与多组已知特征对比
    :param unknown_embedding: 一组未知特征
    :param know_embeddings: 多组已知特征
    :return: 返回两个参数，分别为已知特征中最相似的下标位置，及最大的相似度
    """
    arr = np.sum(unknown_embedding * know_embeddings, axis=1)
    arr2 = np.linalg.norm(unknown_embedding) * np.linalg.norm(know_embeddings, axis=1)
    arr /= arr2

    sim_idx = np.argmax(arr)
    return int(sim_idx), float(arr[sim_idx])


class FaceRecognizer(BaseVisionTask):
    def __init__(self,  model_path: str, thread_num: int = 2, use_gpu: bool = False):
        """
        脸部特征提取
        :param model_path: onnx模型文件路径
        :param thread_num: 线程数量，默认2个线程
        :param use_gpu: 是否使用显卡推理，目前仅支持cuda
        """
        super().__init__(model_path, thread_num, use_gpu)

        self._box_side = 112
        self._arcface_dst = np.float32(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
        self._input_mean = 127.5
        self._input_std = 127.5

    def _preprocess(self, img_obj: Union[str, bytes, np.ndarray], face_5points: np.ndarray) -> np.ndarray:
        """
        人脸对齐
        :param img_obj: 图片对象
        :param face_5points: 脸部的五个关键点位置
        :return: 人脸对齐后的图片
        """
        img = read_image(img_obj)
        m, _ = cv2.estimateAffinePartial2D(face_5points, self._arcface_dst, method=cv2.RANSAC, ransacReprojThreshold=100)
        warped_img = cv2.warpAffine(img, m, (self._box_side, self._box_side))
        return warped_img

    def _preprocess2(self, input_img: np.ndarray) -> np.ndarray:
        """
        图像预处理第二步
        :param input_img: 人脸对齐后的图片
        :return: 可供模型输入的数据
        """
        input_tensor = input_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        input_tensor -= self._input_mean
        input_tensor /= self._input_std
        return input_tensor[np.newaxis, ...]

    def forward(self, img_obj: Union[str, bytes, np.ndarray], face_5points: np.ndarray) -> np.ndarray:
        """
        输入图像得到人脸特征数据
        :param img_obj: 图片对象
        :param face_5points: 脸部的五个关键点位置
        :return: 人脸特征数据
        """
        input_img = self._preprocess(img_obj, face_5points)
        input_tensor = self._preprocess2(input_img)

        output_tensor = self.model.run(None, {self.input_name: input_tensor})[0][0]
        return output_tensor
