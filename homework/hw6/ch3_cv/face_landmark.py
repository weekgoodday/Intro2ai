# 人脸特征点提取器的预训练模型

import onnxruntime
import numpy as np
import cv2

class LandmarksExtractor(object):
    def __init__(self, onnx_path):
        # 加载onnx模型，session是一个会话，用于执行模型
        self.sess = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self._input_name = self.sess.get_inputs()[0].name


    def pre_process(self, img, boxes):
        face_imgs = []
        sizes = []
        for box in boxes:
            cropped = crop(img, box)

            h, w = cropped.shape[:2]
            sizes.append((w, h))

            face_img = _transform(cropped)
            face_imgs.append(face_img)
        
        return face_imgs, sizes

    def post_process(self,outputs,rectangles,sizes):
        """
        Returns results (n, 68,2).
        Args:
            outputs: (n, 136) Net outputs ndarrays 
            rectangles: (n, 4) Rectangles ndarrays
            sizes: (n, 2) Sizes ndarrays
        Returns:
            lms: (n, 68,2) Landmarks ndarrays 
        """
        # output=outputs[0]
        lms = []
        for output, rectangle, (width,height) in zip(outputs, rectangles, sizes):
            points = output.reshape(-1, 2) * (width,height)
            for i in range(len(points)):
                points[i] += (rectangle[0], rectangle[1])

            lms.append(points)
        return lms

    def predict(self, img, boxes):
        """
        Returns results (n, 68,2).
        Args:
            img: image ndarray (h,w,3) 
            boxes: (n, 4) Rectangles ndarrays
        Returns:
            lms: (n, 68,2) Landmarks ndarrays 
        """
        face_imgs, sizes = self.pre_process(img, boxes)
        face_imgs = np.array(face_imgs)
        outputs = self.sess.run(None, {self._input_name: face_imgs})[0]
        lms = self.post_process(outputs, boxes, sizes)
        return lms

    
def _transform(img):
    """
    Returns pre-processed ndarray (h,w,3).
    Args:
        data_raw: raw data ndarray (h,w,3) 
    Returns:
        pre-processed ndarray (3,h,w)
    """
    data_raw = img
    data_raw = cv2.resize(data_raw, (112, 112),
                          interpolation=cv2.INTER_LINEAR)
    data_raw = cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB)
    data_raw = data_raw.astype(np.float32)
    data_raw = data_raw/255.
    data_infer = np.transpose(data_raw, [2, 0, 1])  # [None]
    return data_infer


def crop(image, rectangle):
    """
    Returns cropped image.
    Args:
        image: Bitmap
        rectangle: Rectangle

    Returns:
        Bitmap
    """
    h, w, _ = image.shape

    x0 = max(min(w, rectangle[0]), 0)
    x1 = max(min(w, rectangle[2]), 0)
    y0 = max(min(h, rectangle[1]), 0)
    y1 = max(min(h, rectangle[3]), 0)

    num = image[y0:y1, x0:x1]
    return num

if __name__ == "__main__":
    boxes = np.array([[ 722,   88,  786,  152],
       [ 136,  186,  189,  238],
       [ 342,  145,  409,  212],
       [ 895,  153,  929,  186],
       [ 967,  156, 1003,  191],
       [   8,  249,   50,  291],
       [ 258,  219,  301,  261],
       [ 938,   91,  962,  115],
       [ 920,  191,  953,  223],
       [ 864,  205,  901,  242]])

    img_path = "data/wild.jpg"
    img = cv2.imread(img_path)

    lm_extractor = LandmarksExtractor("weights/landmarks_68_pfld_dy_sim.onnx")
    landmarks = lm_extractor.predict(img, boxes)

    for lm in landmarks:
        for x, y in lm:
            cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0), -1)

    cv2.imshow("img", img)
    cv2.waitKey()