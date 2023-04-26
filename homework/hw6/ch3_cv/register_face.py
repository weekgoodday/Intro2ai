# 注册文件夹中的人脸特征

import json
import os
import traceback

import numpy as np


import onnxruntime

from face_detector import Detector
from face_landmark import LandmarksExtractor
from arcface import ArcFace
from arcface import l2_norm, face_distance, sub_feature

from facial_utils import FacialInfo, parse_filename, remove_old

from facial_utils import imread


def worker_dir(path_dir, detector: Detector, lm_extractor: LandmarksExtractor, arcface: ArcFace, over_write=False):
    '''
    计算一个人的资料文件夹中所有照片的人脸特征

    '''
    facial_info = FacialInfo() # 人脸信息类

    print('Registing dir:{}'.format(path_dir))
    info_path = os.path.join(path_dir, 'info.json')

    empty_flag_path = os.path.join(path_dir, 'empty.flag') # 空文件标志
    nb_imgs = 0
    name_celeba, id_douban, gender_id = parse_filename(path_dir)

    if ((not os.path.exists(info_path)) or over_write):
        if os.path.exists(info_path):
            os.remove(info_path)
        path_list = [os.path.join(path_dir, fn) for fn in os.listdir(path_dir)]

        feature_list = []
        for path in path_list:
            try:
                if path[-4:] in ['flag', 'json']:
                    continue
                # print('Working on img:{}'.format(path))
                img_src = imread(path)
                if img_src is None:
                    continue
                rectangles, _ = detector.predict(img_src)

                if not len(rectangles) == 1:# 图片里只能有一个人脸
                    # print("More than 1 face or no face in img, PASS")
                    continue
                
                
                lm = lm_extractor.predict(img_src, rectangles)[0] # 人脸关键点
                feature_vec = arcface.predict(img_src, [lm])[0] # 人脸特征
                feature_list.append(feature_vec) # 添加到人脸特征列表
                nb_imgs += 1

            except KeyboardInterrupt:
                traceback.print_exc()
                quit()

            except:
                traceback.print_exc()
                continue

        if len(feature_list):
            feature_list = np.asarray(feature_list)
            mean_feature = np.mean(feature_list, axis=0) # 人脸特征均值

            feature_list, mean_feature = sub_feature(feature_list) #根据人脸特征向量的均值，剔除距离均值较远的人脸特征向量
            print("Computed facial feature vector from {} images from directory {}, got {} faces.\nActor's name is {}.".format(
                nb_imgs, path_dir, len(feature_list), name_celeba))

            # 人脸信息字典
            info_dict = {"name": name_celeba, "id": id_douban, "gender_id": gender_id,
                         "feature_vector": mean_feature.tolist(), "feature_list": feature_list.tolist()}

            facial_info.update(info_dict)
            facial_info.save_info(path_json=info_path)

            flag_empty = False
            if os.path.exists(empty_flag_path):
                os.remove(empty_flag_path)
        else:
            flag_empty = True
            with open(empty_flag_path, 'w') as fp:
                fp.write('e')
            mean_feature = None
    elif os.path.exists(empty_flag_path):
        flag_empty = True
        mean_feature = None
    else:
        print('Loading info from {}'.format(info_path))
        info_dict = facial_info.load_info(info_path)
        flag_empty = False
        if os.path.exists(empty_flag_path):
            os.remove(empty_flag_path)

    return facial_info.info_dict, flag_empty


def register_all(root_dir, out_path, detector: Detector, lm_extractor: LandmarksExtractor, arcface: ArcFace, over_write=False):
    '''
    注册文件夹中的人脸特征
    args:
        root_dir: 文件夹路径
        out_path: 输出路径
        detector: 人脸检测器
        lm_extractor: 人脸关键点检测器
        arcface: 人脸特征提取器
        over_write: 是否覆盖已有的特征
    '''
    if over_write:
        remove_old(root_dir)

    pathes_dir = [path_dir for path_dir in [os.path.join(root_dir, name_dir) for name_dir in os.listdir(root_dir)] if os.path.isdir(path_dir)]

    results = []

    print('Registing {} by serial processing'.format(root_dir))
    results = [worker_dir(path_dir, detector, lm_extractor, arcface, over_write) for path_dir in pathes_dir]

    db_dict = {}

    for (info_dict, flag_empty) in results:
        if not flag_empty:
            db_dict.setdefault(info_dict["id"], info_dict)

    with open(out_path, "w") as fp:
        json.dump(db_dict, fp)

    return db_dict


if __name__ == "__main__":
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    detector = Detector("weights/face_detector_640_dy_sim.onnx", input_size=(640, 480), top_k=16)
    lm_extractor = LandmarksExtractor("weights/landmarks_68_pfld_dy_sim.onnx")
    arcface = ArcFace("weights/arc_mbv2_ccrop_sim.onnx")

    root_dir = "data/imgs_celebrity"  # save face image
    out_path = "t.json"
    register_all(root_dir, out_path, detector, lm_extractor, arcface, use_par=False, over_write=True)
