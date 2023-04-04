# 用于人脸检测的预训练模型

import onnxruntime
import numpy as np
import cv2


class Detector(object):

    def __init__(self, onnx_path, input_size=(640, 480), confidenceThreshold=0.95, nmsThreshold=0.5, top_k=16):
        self.input_size = input_size
        self.confidenceThreshold = confidenceThreshold
        self.nmsThreshold = nmsThreshold
        self.top_k = top_k
        self.onnx_path = onnx_path
        # 加载onnx模型，session是一个会话，用于执行模型
        self.session = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self._input_name = self.session.get_inputs()[0].name

    def pre_process(self, img):
        '''
        图像预处理
        :param img: BGR image
        :return: preprocessed image
        '''
        img = cv2.resize(img, self.input_size,
                        interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img-127.)/128.
        img_infer = np.transpose(img, [2, 0, 1])  # [None]

        return img_infer

    def predict(self, img):
        '''
        预测
        :param img: BGR image
        :return: boxes, confidences
        '''
        width, height = img.shape[1], img.shape[0]

        img_infer = self.pre_process(img)
        img_infer = np.expand_dims(img_infer, axis=0)
        img_infer = img_infer.astype(np.float32)

        # 执行模型推理
        boxes_batch, confidences_batch = self.session.run(None, {self._input_name: img_infer})
        boxes_raw, confidences_raw = boxes_batch[0], confidences_batch[0]# 一个batch中只有一张图片

        boxes, confidences = self.post_process(boxes_raw, confidences_raw, width, height, self.confidenceThreshold, self.nmsThreshold, self.top_k)
        return boxes, confidences



    def post_process(self,  boxes, confidences, width, height, confidenceThreshold, nmsThreshold, top_k):
        '''
        后处理
        :param boxes: boxes
        :param confidences: confidences
        :param width: image width
        :param height: image height
        :param confidenceThreshold: confidence threshold
        :param nmsThreshold: nms threshold
        :param top_k: top k
        :return: boxes, confidences
        '''
        boxes, confidences = _parse_result(width, height, boxes, confidences, confidenceThreshold, nmsThreshold, top_k)
        return boxes, confidences


def _parse_result(width,height, boxes, confidences, prob_threshold, iou_threshold=0.5, top_k=5):
    """
    Selects boxes that contain human faces.
    Args:
        width: original image width
        height: original image height
        boxes (N, K, 4): an array of boxes.
        confidences (N, K, 2): an array of probabilities.
        prob_threshold: a threshold used to filter boxes by the probability.
        iou_threshold: a threshold used in non maximum suppression.
        top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'.
    Returns:
        boxes (N, K, 4): an array of boxes kept
        probs (N, K): an array of probabilities for each boxes being in corresponding labels
    """
    picked_box_probs = []

    probs = confidences[:, 1]
    mask = probs > prob_threshold
    probs = probs[mask]

    if len(probs) == 0:
        return np.array([]), np.array([])

    subset_boxes = boxes[mask, :]
    box_probs = np.concatenate(
        [subset_boxes, probs.reshape(-1, 1)], axis=1)
    box_probs = _hard_nms(box_probs,
                        iou_threshold=iou_threshold,
                        top_k=top_k,
                        )
    picked_box_probs.append(box_probs)

    if not picked_box_probs:
        return np.array([]), np.array([])

    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    picked_boxes= ToBoxN(picked_box_probs[:, :4].astype(np.int32))
    picked_probs = picked_box_probs[:, 4]

    return picked_boxes, picked_probs


def _area_of(left_top, right_bottom):
    """
    Computes the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def _iou_of(boxes0, boxes1, eps=1e-5):
    """
    Returns intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = _area_of(overlap_left_top, overlap_right_bottom)
    area0 = _area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = _area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def _hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Performs hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = _iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def ToBoxN(rectangle_N):
    """
    Returns rectangle scaled to box.
    Args:
        rectangle: Rectangle
    Returns:
        Rectangle
    """
    width = rectangle_N[:, 2] - rectangle_N[:, 0]
    height = rectangle_N[:, 3] - rectangle_N[:, 1]
    m = np.max([width, height], axis=0)
    dx = ((m - width)/2).astype("int32")
    dy = ((m - height)/2).astype("int32")

    rectangle_N[:, 0] -= dx
    rectangle_N[:, 1] -= dy
    rectangle_N[:, 2] += dx
    rectangle_N[:, 3] += dy
    return rectangle_N

if __name__ == "__main__":
    face_detector = Detector("weights/face_detector_640_dy_sim.onnx")

    img_path = "data/wild.jpg"
    img = cv2.imread(img_path)

    # Detect faces
    boxes, confidences = face_detector.predict(img)

    for rect in boxes:
        x1, y1, x2, y2 = rect
        # Draw rectangles
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
