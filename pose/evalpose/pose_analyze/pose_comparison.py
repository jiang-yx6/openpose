# pose_comparison.py
import numpy as np


def readme_normalize_landmarks(landmarks):
    """
    使用 README 描述的方法对原始 landmarks 进行归一化：
    输入 landmarks：列表，每个元素格式为 [index, x, y]
    输出：归一化后的关键点列表，每个元素为 [norm_x, norm_y]，归一化到 [0, 1]
    """
    xs = [lm[1] for lm in landmarks]
    ys = [lm[2] for lm in landmarks]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    range_x = max_x - min_x
    range_y = max_y - min_y
    normalized = []
    for lm in landmarks:
        norm_x = (lm[1] - min_x) / range_x if range_x != 0 else 0
        norm_y = (lm[2] - min_y) / range_y if range_y != 0 else 0
        normalized.append([norm_x, norm_y])
    return normalized


def flatten_pose(frame, use_readme_norm=True):
    """
    将单帧姿态数据展平成 1D 特征向量。
    如果 use_readme_norm 为 True，则使用 README 的归一化方法对原始 landmarks 进行归一化；
    否则使用 frame 中的 'norm_landmarks'（由肩宽归一化）。
    同时附加角度信息（不归一化）。
    """
    if use_readme_norm and "landmarks" in frame:
        normalized_landmarks = readme_normalize_landmarks(frame["landmarks"])
    else:
        normalized_landmarks = frame.get("norm_landmarks", [])

    if not normalized_landmarks:
        return None
    landmarks_flat = np.array(normalized_landmarks).flatten()
    # 角度数据保持原样（例如 0~180°）
    angles = np.array(list(frame.get("angles", {}).values()))
    feature_vector = np.concatenate([landmarks_flat, angles])
    return feature_vector


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return cos_sim


def score_cos_sim(pose1, pose2):
    """
    使用 README 归一化后的特征，计算两个姿态帧之间的余弦相似度，
    并返回 0～100 之间的得分。
    """
    vec1 = flatten_pose(pose1, use_readme_norm=True)
    vec2 = flatten_pose(pose2, use_readme_norm=True)
    if vec1 is None or vec2 is None:
        return 0.0
    cos_sim = cosine_similarity(vec1, vec2)
    cos_sim = max(cos_sim, 0)
    score = cos_sim * 100
    return score


def weight_match_l1(pose1, pose2, weights=None):
    """
    使用 L1 范数计算加权匹配得分，返回 0～100 之间的得分。
    使用 README 的归一化方法进行归一化。
    为降低绝对差异带来的惩罚，先将 L1 差异除以特征向量长度。
    """
    vec1 = flatten_pose(pose1, use_readme_norm=True)
    vec2 = flatten_pose(pose2, use_readme_norm=True)
    if vec1 is None or vec2 is None:
        return 0.0
    if weights is not None:
        weights = np.array(weights)
        if weights.shape[0] != vec1.shape[0]:
            raise ValueError("Weights dimension does not match feature vector dimension.")
        diff = np.abs(vec1 - vec2) * weights
    else:
        diff = np.abs(vec1 - vec2)
    L1_diff = np.sum(diff)
    feature_length = vec1.shape[0]
    normalized_L1 = L1_diff / feature_length
    score = 100 * (1 / (1 + normalized_L1))
    return score


def weight_match_l2(pose1, pose2, weights=None):
    """
    使用 L2 范数计算加权匹配得分，返回 0～100 之间的得分。
    使用 README 的归一化方法进行归一化。
    先将 L2 差异除以特征向量长度，再计算得分。
    """
    vec1 = flatten_pose(pose1, use_readme_norm=True)
    vec2 = flatten_pose(pose2, use_readme_norm=True)
    if vec1 is None or vec2 is None:
        return 0.0
    if weights is not None:
        weights = np.array(weights)
        if weights.shape[0] != vec1.shape[0]:
            raise ValueError("Weights dimension does not match feature vector dimension.")
        diff = (vec1 - vec2) * weights
    else:
        diff = vec1 - vec2
    L2_diff = np.linalg.norm(diff)
    feature_length = vec1.shape[0]
    normalized_L2 = L2_diff / feature_length
    score = 100 * (1 / (1 + normalized_L2))
    return score


def dtw_compare(seq1, seq2, scaler=None):
    """
    利用 DTW 比较两个视频序列的姿态数据，返回 DTW 距离及归一化后的相似度得分（0～100）。
    使用 README 的归一化方法对每一帧进行归一化，再计算特征向量差异。
    """
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    features1 = []
    features2 = []
    for frame in seq1:
        vec = flatten_pose(frame, use_readme_norm=True)
        if vec is not None:
            features1.append(vec)
    for frame in seq2:
        vec = flatten_pose(frame, use_readme_norm=True)
        if vec is not None:
            features2.append(vec)
    if len(features1) == 0 or len(features2) == 0:
        return None, 0.0
    features1 = np.array(features1)
    features2 = np.array(features2)
    if scaler is not None:
        features1 = scaler.fit_transform(features1)
        features2 = scaler.transform(features2)
    distance, _ = fastdtw(features1, features2, dist=euclidean)
    max_length = max(len(features1), len(features2))
    normalized_distance = distance / max_length
    similarity = 100 * (1 / (1 + normalized_distance))
    return distance, similarity
