import cv2 
import numpy as np

#PIL用于绘制中文
from PIL import Image
from PIL import ImageDraw, ImageFont

from face_detector import Detector
from face_landmark import LandmarksExtractor
from arcface import ArcFace

from register_face import register_all
from facial_utils import FacialDB

face_detector = Detector("weights/face_detector_640_dy_sim.onnx")
lm_extractor = LandmarksExtractor("weights/landmarks_68_pfld_dy_sim.onnx")
arcface = ArcFace("weights/arc_mbv2_ccrop_sim.onnx")


img_path = "data/group_16.jpg"
img_wild = cv2.imread(img_path)

# Detect faces
boxes, confidences = face_detector.predict(img_wild)

# Detect landmarks
landmarks = lm_extractor.predict(img_wild, boxes)


embs = arcface.predict(img_wild, landmarks)


# Register all faces
dir_info = "data/my_imgs"
path_json_tmp = "t.json"

db_dict = register_all(dir_info, path_json_tmp,
                        face_detector, lm_extractor, arcface, over_write=True)
facial_db = FacialDB(db_dict=db_dict)# 实例化FacialDB人脸数据库类

inds, knowns, dists_max = facial_db.query_N2N(embs)


for rect, ind, known, score, landmark in zip(boxes, inds, knowns, dists_max, landmarks):
        rect = list(map(int, rect))
        cv2.rectangle(img_wild, (rect[0], rect[1]),
                      (rect[2], rect[3]), (0, 0, 255), 2)
        cx = rect[0]
        cy = rect[1] + 12

        if score > 0.7: #置信度大于0.7
            name = facial_db.ind2name[ind] # 从数据库中获取名字

            # 绘制中文
            image = Image.fromarray(cv2.cvtColor(img_wild, cv2.COLOR_BGR2RGB)) # PIL读取的格式为RGB，而OpenCV读取的格式为BGR
            draw = ImageDraw.Draw(image) # 绘画对象
            fontText = ImageFont.truetype(
                "data/fonts/SourceHanSansHWSC-Regular.otf", 18, encoding="utf-8") # 字体
            draw.text((cx, cy-32), name, font=fontText, fill=(0, 255, 0)) # 绘制文本
            img_wild = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)   # PIL绘制的图像格式为RGB，而OpenCV绘制的图像格式为BGR

        for (x, y) in landmark:
            cv2.circle(img_wild, (int(x), int(y)), 2, (255, 255, 0), -1)
        
# cv2.imshow("img", img_wild)
# cv2.waitKey()
cv2.imwrite("data/group_result_16.jpg", img_wild)