# visualization.py
import cv2
import numpy as np
import os

def draw_bone(img, landmarks, connections, color=(0, 255, 0)):
    """
    绘制骨架连接
    :param img: 图像
    :param landmarks: 关节点坐标列表
    :param connections: 连接关节点的索引对列表
    :param color: 绘制颜色
    """
    for connection in connections:
        x1, y1 = landmarks[connection[0]]
        x2, y2 = landmarks[connection[1]]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

def generate_video_with_selected_frames(std_video, pat_video, dtw_result, output_video_path, video_path_pat, stages, config, save_lowest_scores=True):
    """
    生成包含骨架对比的输出视频，同时保存最低得分帧的图像
    :param std_video: 标准视频的帧数据序列
    :param pat_video: 患者视频的帧数据序列
    :param dtw_result: DTW 对齐结果
    :param output_video_path: 输出视频保存路径
    :param video_path_pat: 患者视频文件路径
    :param stages: 动作阶段列表
    :param config: 配置对象，包含KEY_ANGLES等参数
    :param save_lowest_scores: 是否保存最低得分帧
    """
    from .evaluation import select_lowest_score_frames
    lowest_score_frames = select_lowest_score_frames(dtw_result, stages)

    cap_pat = cv2.VideoCapture(video_path_pat)
    if not cap_pat.isOpened():
        raise ValueError("无法打开视频文件")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap_pat.get(cv2.CAP_PROP_FPS)
    width = int(cap_pat.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_pat.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 构建患者帧索引到标准帧索引的映射
    pat_to_std = {}
    for std_idx, pat_idx in dtw_result['alignment_path']:
        if pat_idx not in pat_to_std:
            pat_to_std[pat_idx] = std_idx

    frame_idx = 0
    idx = 1
    
    while cap_pat.isOpened():
        success_pat, img_pat = cap_pat.read()
        if not success_pat:
            break

        # 超出frame序列范围检查
        if frame_idx >= len(pat_video):
            break
            
        pat_frame = pat_video[frame_idx]
        # 尝试获取对应的标准帧索引
        if frame_idx in pat_to_std:
            std_frame_idx = pat_to_std[frame_idx]
        else:
            available_indices = sorted(pat_to_std.keys())
            candidates = [idx for idx in available_indices if idx <= frame_idx]
            if candidates:
                std_frame_idx = pat_to_std[candidates[-1]]
            elif available_indices:
                std_frame_idx = pat_to_std[available_indices[0]]
            else:
                std_frame_idx = 0  # 默认取第0帧

        if std_frame_idx >= len(std_video):
            std_frame_idx = len(std_video) - 1  # 避免越界
        std_frame = std_video[std_frame_idx]

        pat_landmarks = [(lm[1], lm[2]) for lm in pat_frame['landmarks']]

        # 绘制患者骨架（红色）
        for angle_name, joints in config.KEY_ANGLES.items():
            p1, p2, p3 = joints
            draw_bone(img_pat, pat_landmarks, [(p1, p2), (p2, p3)], color=(0, 0, 255))

        pat_shoulder_left = np.array(pat_frame['landmarks'][11][1:])
        std_landmarks = [(lm[1], lm[2]) for lm in std_frame['landmarks']]

        # 将标准骨架平移对齐到患者的左肩位置
        std_shoulder_left = np.array(std_frame['landmarks'][11][1:])
        translation_vector = pat_shoulder_left - std_shoulder_left
        std_landmarks_translated = []
        for x, y in std_landmarks:
            x_translated = x + translation_vector[0]
            y_translated = y + translation_vector[1]
            std_landmarks_translated.append((x_translated, y_translated))

        # 绘制标准骨架（绿色）
        for angle_name, joints in config.KEY_ANGLES.items():
            p1, p2, p3 = joints
            draw_bone(img_pat, std_landmarks_translated, [(p1, p2), (p2, p3)], color=(0, 255, 0))

        # 保存最低得分帧
        if save_lowest_scores and frame_idx in [frame[0] for frame in lowest_score_frames]:
            img_name = os.path.join(os.path.dirname(output_video_path), f'patient_frame_{idx}.jpg')
            idx += 1
            cv2.imwrite(img_name, img_pat)
            print(f"保存最低得分的患者帧：{img_name}")

        out.write(img_pat)
        frame_idx += 1

    cap_pat.release()
    out.release()
    
    return output_video_path