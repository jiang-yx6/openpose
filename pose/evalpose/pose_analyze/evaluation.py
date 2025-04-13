# evaluation.py
def detect_action_stages(sequence, angle_threshold=5):
    """
    根据关键点角度变化自动识别动作阶段。
    :param sequence: 每帧的姿势特征序列
    :param angle_threshold: 判断动作变化的角度阈值
    :return: 阶段列表，每个阶段以 (起始帧, 结束帧) 表示
    """
    stages = []
    prev_angles = None
    stage_start = 0

    for i, frame in enumerate(sequence):
        angles = frame['angles']
        if prev_angles:
            angle_diff = {joint: abs(angles[joint] - prev_angles.get(joint, 0)) for joint in angles}
            if any(diff > angle_threshold for diff in angle_diff.values()):
                stage_end = i
                stages.append((stage_start, stage_end))
                stage_start = i
        prev_angles = angles
    stages.append((stage_start, len(sequence)-1))
    return stages

def select_lowest_score_frames(dtw_result, stages, max_frames=3):
    """
    在识别的阶段中选取得分最低的帧。
    :param dtw_result: DTW 对齐结果，包含各帧得分
    :param stages: 动作阶段列表
    :param max_frames: 最多选择的帧数
    :return: 选定的最低得分帧列表
    """
    frame_scores = dtw_result['frame_scores']
    lowest_frames = []
    for stage_start, stage_end in stages:
        stage_scores = frame_scores[stage_start:stage_end+1]
        lowest_frames.append(sorted(stage_scores, key=lambda x: x[1])[0])
    if len(stages) > max_frames:
        lowest_frames = sorted(lowest_frames, key=lambda x: x[1])[:max_frames]
    return lowest_frames
