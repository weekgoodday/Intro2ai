# 用于人脸特征提取的ArcFace模型

import onnxruntime
import numpy as np
import cv2

MEAN_PTS_5 = np.array([[0.34191607, 0.46157411],
                       [0.65653392, 0.45983393],
                       [0.500225, 0.64050538],
                       [0.3709759, 0.82469198],
                       [0.63151697, 0.82325091]])

INDS_68_5 = [36, 45, 30, 48, 54]


class ArcFace(object):
    def __init__(self, onnx_path, input_size=(112, 112)):
        # 加载onnx模型，session是一个会话，用于执行模型
        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_size = input_size
        self._input_name = self.session.get_inputs()[0].name

    def pre_process(self, img, landmarks):
        '''
        :param img: 输入图像, BGR格式
        :param landmarks: 人脸关键点, (n,68,2)的ndarray
        '''
        face_imgs = []
        for lm in landmarks:
            face_imgs.append(_align_face(img, lm))

        return np.ascontiguousarray(face_imgs)

    def post_process(self, outputs):
        return [l2_norm(emb) for emb in outputs]

    def predict(self, img, landmarks):
        '''
        :param img: 输入图像, BGR格式
        :param landmarks: 人脸关键点, (n,68,2)的ndarray
        '''
        face_imgs = self.pre_process(img, landmarks)
        outputs = self.session.run(None, {self._input_name: face_imgs})[0]
        return self.post_process(outputs)


def _align_face(img, landm):
    '''
    人脸对齐,根据特征点坐标将人脸图片旋转平移到正中间
    '''
    if not landm is None:
        mat = get_transform_mat(
            landm[INDS_68_5].reshape(5, 2), MEAN_PTS_5, 112)
        face_img = warp_img(mat, img, (112, 112)).astype(np.float32)/255.0
        # face_imgs.append(face_img)
    else:
        face_img = cv2.resize(
            img, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    return face_img


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def get_transform_mat(image_landmarks, mean_pts, output_size=112, scale=1.0):
    '''
    Get affine transform matrix between image landmarks and mean landmarks
    '''
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array(image_landmarks)
    padding = 1  # (output_size / 64) * 1

    mat = umeyama(image_landmarks, mean_pts, True)[0:2]
    mat = mat * (output_size - 2 * padding)
    mat[:, 2] += padding
    mat *= (1 / scale)
    mat[:, 2] += -output_size*(((1 / scale) - 1.0) / 2)

    return mat


def transform_points(points, mat, invert=False):
    '''
    Transform points with affine matrix
    '''
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points


def warp_img(mat, img, dshape=(112, 112), invert=False):
    '''
    Apply affine transform to image
    '''
    if invert:
        M = cv2.invertAffineTransform(mat)
    else:
        M = mat
    warped = cv2.warpAffine(img, M, dshape, cv2.INTER_LANCZOS4)
    return warped


def l2_norm(x, axis=-1):
    '''
    L2 normalize embeddings
    '''
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def face_distance(known_face_encoding, face_encoding_to_check):
    '''
    Compute distance between two face encodings
    '''
    fl = np.asarray(known_face_encoding)
    return np.dot(fl, face_encoding_to_check)


def face_identify(known_face_encoding, face_encoding_to_check, tolerance=0.6):
    '''
    Identify face encoding

    '''
    distance = face_distance(known_face_encoding, face_encoding_to_check)

    argmax = np.argmax(distance)
    d_min = distance[argmax]

    if distance[argmax] < tolerance:
        index = -1
        is_known = False
    else:
        index = argmax
        is_known = True
    return is_known, index, d_min


def sub_feature(feature_list, rate=0.9):
    '''
    根据人脸特征向量的均值，剔除距离均值较远的人脸特征向量
    '''
    feature_list = np.asarray(feature_list)
    mean_feature = np.mean(feature_list, axis=0)

    nb_feature = int(rate*len(feature_list))
    if nb_feature:
        dists = face_distance(feature_list, mean_feature)

        sub_feature_list = feature_list[np.argsort(dists)[::-1][:nb_feature]]
        mean_feature = l2_norm(np.mean(sub_feature_list, axis=0))
        return sub_feature_list, mean_feature
    else:
        return feature_list.copy(), feature_list[0].copy()
