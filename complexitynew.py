import cv2
import numpy as np
import os
import csv
import argparse


def compute_flow_magnitude_direction(flow):
    """
    输入光流 flow (H, W, 2)
    返回幅值 mag 和方向 ang (弧度)
    """
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)
    return mag, ang


def compute_frame_complexity(mag, ang_diff_hist, alpha=0.5):
    """
    mag: 当前帧光流幅值
    ang_diff_hist: 当前帧光流差方向的直方图
    alpha: 幅值与方向复杂度权重
    """
    mag_std = np.std(mag)
    ang_std = np.std(ang_diff_hist)
    return alpha * mag_std + (1 - alpha) * ang_std


def compute_video_complexity(ref_path, dis_path, alpha=0.5, bins=30, min_mag_threshold=1e-3):
    """
    计算失真视频相对于参考视频的复杂度
    输出每帧复杂度列表和总体复杂度
    """
    cap_ref = cv2.VideoCapture(ref_path)
    cap_dis = cv2.VideoCapture(dis_path)

    prev_ref, prev_dis = None, None
    prev_flow_dis = None
    frame_complexities = []

    while True:
        ret_ref, frame_ref = cap_ref.read()
        ret_dis, frame_dis = cap_dis.read()
        if not ret_ref or not ret_dis:
            break

        gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
        gray_dis = cv2.cvtColor(frame_dis, cv2.COLOR_BGR2GRAY)

        if prev_ref is not None and prev_dis is not None:
            # 计算光流
            flow_dis = cv2.calcOpticalFlowFarneback(prev_dis, gray_dis, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_ref = cv2.calcOpticalFlowFarneback(prev_ref, gray_ref, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 幅值复杂度（使用差分光流幅值）
            mag, _ = compute_flow_magnitude_direction(flow_dis - flow_ref)

            # 方向一致性：基于差分光流
            if prev_flow_dis is not None:
                flow_diff = flow_dis - prev_flow_dis
                mag_diff, ang_diff = compute_flow_magnitude_direction(flow_diff)

                # 可选：只考虑幅值大于阈值的像素，减少噪声
                mask = mag_diff > min_mag_threshold
                ang_diff_masked = ang_diff[mask]

                # 方向直方图
                if len(ang_diff_masked) > 0:
                    hist, _ = np.histogram(ang_diff_masked, bins=bins, range=(0, 2 * np.pi))
                    ang_std = np.std(hist)
                else:
                    ang_std = 0.0

                c = alpha * np.std(mag) + (1 - alpha) * ang_std
                frame_complexities.append(c)

            prev_flow_dis = flow_dis

        prev_ref, prev_dis = gray_ref, gray_dis

    cap_ref.release()
    cap_dis.release()

    if len(frame_complexities) == 0:
        return [], 0.0

    mean_c = np.mean(frame_complexities)
    std_c = np.std(frame_complexities)
    overall = mean_c + std_c

    return frame_complexities, overall


def process_csv(input_csv, output_csv, alpha=0.5):
    """
    从输入CSV读取视频对，计算复杂度并输出结果
    """
    results = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            dis_path, ref_path = row[0].strip(), row[1].strip()
            if not (os.path.exists(dis_path) and os.path.exists(ref_path)):
                print(f"[警告] 文件不存在：{dis_path} 或 {ref_path}")
                continue

            print(f"正在处理: {os.path.basename(dis_path)}")
            frame_cs, overall_c = compute_video_complexity(ref_path, dis_path, alpha)
            frame_str = " ".join([f"{v:.6f}" for v in frame_cs])
            results.append([os.path.basename(dis_path), frame_str, f"{overall_c:.6f}"])

    # 输出到 CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "frame_complexities", "overall_complexity"])
        writer.writerows(results)

    print(f"\n✅ 已保存结果到 {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute optical flow difference complexity for video pairs.")
    parser.add_argument("--input_csv", type=str, required=True, help="输入CSV文件，包含每行的失真视频路径,参考视频路径")
    parser.add_argument("--output_csv", type=str, default="complexity_results.csv", help="输出CSV文件路径")
    parser.add_argument("--alpha", type=float, default=0.5, help="幅值与方向复杂度权重参数α")
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv, args.alpha)
