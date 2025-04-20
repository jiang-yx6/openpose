# main.py
import numpy as np
import json
from pose_detector import VideoAnalyzer
from action_comparator import ActionComparator
from evaluation import detect_action_stages
from visualization import generate_video_with_selected_frames
from video_stretch import stretch_videos_to_same_length
from config_service import get_config_class

def main():
    # Use default config
    Config = get_config_class(None)
    print(f"Using configuration: {Config.DESCRIPTION if hasattr(Config, 'DESCRIPTION') else 'Default'}")
    print(f"Key angles: {Config.KEY_ANGLES}")
    print(f"Normalization joints: {Config.NORMALIZATION_JOINTS}")
    
    # Initialize analyzer with the dynamic config
    analyzer = VideoAnalyzer(config=Config)

    # 假设原视频路径如下，输出视频路径如下：
    video1 = "PoseVideos/standard.mp4"
    video2 = "PoseVideos/demo2.mp4"
    #output1 = "PoseVideos/standard_stretched.avi"
    #output2 = "PoseVideos/demo2_stretched.avi"

    #target_frames = stretch_videos_to_same_length(video1, video2, output1, output2, fps=30)
    #print("拉伸后目标帧数：", target_frames)

    # 处理标准视频与患者视频（路径保持原有）
    std_video = analyzer.process_video("PoseVideos/standard.mp4")
    pat_video = analyzer.process_video("PoseVideos/demo2.mp4")

    # 使用 ActionComparator 进行动作比较（DTW 对齐及帧匹配评分）
    comparator = ActionComparator(std_video, pat_video)
    result = comparator.compare_sequences()

    speed_variation = comparator.analyze_speed_variation(result['alignment_path'])
    print("动作速度分析：", speed_variation)

    # 假设 result['alignment_path'] 已经计算得到 DTW 对齐路径
    angle_result = comparator.compute_alignment_angle_score(result['alignment_path'])
    print("对齐角度评分：", angle_result)

    # 新增：计算对齐帧比率（相邻差值不变的比例）
    alignment_ratio = comparator.compute_alignment_ratio(result['alignment_path'])
    alignment_ratio_score = alignment_ratio * 100  # 转换为百分制
    print("对齐帧比例：", alignment_ratio_score)

    # 利用 DTW 对齐路径计算时间差方差得分（新增功能）
    dtw_time_result = comparator.dtw_time_variance_score(result['alignment_path'])
    print("DTW 时间差方差结果：", dtw_time_result)

    # 动作阶段识别（如有需要，可用于视频分段）
    stages = detect_action_stages(pat_video)

    # 生成对齐对比视频，输出路径保持不变，传递动态配置
    output_video_path = "PoseVideos/aligned_comparison_video.avi"
    generate_video_with_selected_frames(std_video, pat_video, result, output_video_path, 
                                       "PoseVideos/demo2.mp4", stages, Config)
    print(f"视频已保存：{output_video_path}")

    # 使用原有 DTW 结果计算自定义相似度评分（公式保持不变）
    dtw_distance = result['dtw_distance']
    similarity_score_custom = 250 / (dtw_distance + 125)
    print("DTW 距离：", dtw_distance)
    print(f"动作相似度评分 (自定义公式): {similarity_score_custom:.2%}")

    # 生成对比报告图片（comparison_report.jpg）
    report_img = comparator.generate_report(result)

    # 俯卧撑动作改进建议（保持原有）
    pushup_advice = {
        'left_elbow': "肘关节活动范围不足，建议加大下沉深度（标准角度：90°±10°）",
        'right_elbow': "左右肘不对称，注意保持双臂对称运动",
        'body_angle': "躯干稳定性不足，保持身体成直线（标准角度：170°-180°）",
        'hip_alignment': "臀部下沉不足/过度抬起，保持髋部与肩部平行"
    }
    pushup_results = {}
    for joint, advice in pushup_advice.items():
        try:
            std_avg = np.mean([f['angles'][joint] for f in std_video if f['angles'][joint] > 0])
            pat_avg = np.mean([f['angles'].get(joint, 0) for f in pat_video])
            pushup_results[joint] = {"advice": advice, "std_avg": std_avg, "pat_avg": pat_avg}
            print(f"- {joint}: {advice}\n 标准角度：{std_avg:.1f}° vs 患者角度：{pat_avg:.1f}°")
        except KeyError:
            pushup_results[joint] = {"advice": advice, "std_avg": None, "pat_avg": None}

    # 新增：调用全视频综合评分方法（compare_overall_video），得到整体动作得分
    overall_scores = comparator.compare_overall_video()
    print("全视频综合评分：", overall_scores)

    # 处理 ActionComparator 返回的 result 以便 JSON 序列化
    result_serializable = {}
    for key, value in result.items():
        if key == "alignment_path":
            result_serializable[key] = [list(t) for t in value]
        elif key == "frame_scores":
            result_serializable[key] = [list(t) for t in value]
        elif key in ["aligned_std", "aligned_pat"]:
            converted = []
            for arr in value:
                if hasattr(arr, "tolist"):
                    converted.append(arr.tolist())
                else:
                    converted.append(arr)
            result_serializable[key] = converted
        else:
            result_serializable[key] = value

    # 汇总所有输出到一个字典中（新增 alignment_ratio_score 字段）
    output_data = {
        "aligned_video": output_video_path,
        "comparison_report": report_img,
        "dtw_distance": dtw_distance,
        "similarity_score_custom": similarity_score_custom,
        "alignment_ratio_score": alignment_ratio_score,
        "dtw_time_result": dtw_time_result,
        "action_comparator": result_serializable,
        "overall_video_score": overall_scores,
        "pushup_advice": pushup_results
    }
    # 保存输出结果到 JSON 文件（文件名保持不变）
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("所有输出已保存至 output.json")


if __name__ == "__main__":
    main()
