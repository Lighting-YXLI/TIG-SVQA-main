import cv2
import numpy as np

def calculate_optical_flow(prev_frame, next_frame):
    return cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

def compute_flow_difference(flow1, flow2):
    return flow1 - flow2  # 保留二维方向

def visualize_flow(flow, frame_shape):
    hsv = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.float32)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)

def visualize_flow_difference_colored(flow_diff, frame_shape):
    magnitude = np.sqrt(flow_diff[..., 0]**2 + flow_diff[..., 1]**2)
    angle = np.arctan2(flow_diff[..., 1], flow_diff[..., 0])
    angle = np.mod(angle, 2 * np.pi)

    hsv = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.float32)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)

def normalize_image_255(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val < 1e-5:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def normalize_color_image(image):
    channels = []
    for i in range(3):
        ch = normalize_image_255(image[..., i])
        channels.append(ch)
    return cv2.merge(channels)

def process_videos_with_filters(distorted_video_path, reference_video_path,
                                coarse_video_path, fine_video_path,
                                original_video_path, reference_video_output_path,
                                flow_diff_video_path, distorted_flow_video_path,
                                reference_flow_video_path):
    cap_dis = cv2.VideoCapture(distorted_video_path)
    cap_ref = cv2.VideoCapture(reference_video_path)

    if not cap_dis.isOpened() or not cap_ref.isOpened():
        print("Error: Could not open one or more videos.")
        return

    width = int(cap_dis.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_dis.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_dis.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_coarse = cv2.VideoWriter(coarse_video_path, fourcc, fps, (width, height))
    out_fine = cv2.VideoWriter(fine_video_path, fourcc, fps, (width, height))
    out_original = cv2.VideoWriter(original_video_path, fourcc, fps, (width, height))
    out_ref = cv2.VideoWriter(reference_video_output_path, fourcc, fps, (width, height))
    out_flow_diff = cv2.VideoWriter(flow_diff_video_path, fourcc, fps, (width, height))
    out_dis_flow = cv2.VideoWriter(distorted_flow_video_path, fourcc, fps, (width, height))
    out_ref_flow = cv2.VideoWriter(reference_flow_video_path, fourcc, fps, (width, height))

    ret_dis, prev_dis = cap_dis.read()
    ret_ref, prev_ref = cap_ref.read()

    while ret_dis and ret_ref:
        ret_dis, cur_dis = cap_dis.read()
        ret_ref, cur_ref = cap_ref.read()
        if not ret_dis or not ret_ref:
            break

        gray_dis = cv2.cvtColor(cur_dis, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(cur_ref, cv2.COLOR_BGR2GRAY)
        prev_gray_dis = cv2.cvtColor(prev_dis, cv2.COLOR_BGR2GRAY)
        prev_gray_ref = cv2.cvtColor(prev_ref, cv2.COLOR_BGR2GRAY)

        flow_dis = calculate_optical_flow(prev_gray_dis, gray_dis)
        flow_ref = calculate_optical_flow(prev_gray_ref, gray_ref)
        flow_diff = compute_flow_difference(flow_dis, flow_ref)

        vis_dis_flow = visualize_flow(flow_dis, cur_dis.shape)
        vis_ref_flow = visualize_flow(flow_ref, cur_ref.shape)
        vis_diff_flow = visualize_flow_difference_colored(flow_diff, cur_dis.shape)

        # === 计算光流差幅度并归一化 ===
        magnitude = np.sqrt(flow_diff[..., 0] ** 2 + flow_diff[..., 1] ** 2)
        flow_diff_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        # === 区域分割 ===
        threshold = 0.2
        coarse_mask = (flow_diff_norm >= threshold).astype(np.float32)
        fine_mask = (flow_diff_norm < threshold).astype(np.float32)
        coarse_mask = cv2.GaussianBlur(coarse_mask, (11, 11), 5)
        fine_mask = cv2.GaussianBlur(fine_mask, (11, 11), 5)

        # === 区域权重强调 ===
        coarse_emphasis = flow_diff_norm * coarse_mask
        fine_emphasis = flow_diff_norm * fine_mask

        coarse_weighted_frame = cur_dis.astype(np.float32)
        fine_weighted_frame = cur_dis.astype(np.float32)

        for c in range(3):
            coarse_weighted_frame[..., c] *= (1 + coarse_emphasis)
            fine_weighted_frame[..., c] *= (1 + fine_emphasis)

        coarse_final = normalize_color_image(coarse_weighted_frame)
        fine_final = normalize_color_image(fine_weighted_frame)

        out_coarse.write(coarse_final)
        out_fine.write(fine_final)
        out_original.write(cur_dis)
        out_ref.write(cur_ref)
        out_flow_diff.write(vis_diff_flow)
        out_dis_flow.write(vis_dis_flow)
        out_ref_flow.write(vis_ref_flow)

        prev_dis = cur_dis
        prev_ref = cur_ref

    cap_dis.release()
    cap_ref.release()
    out_coarse.release()
    out_fine.release()
    out_original.release()
    out_ref.release()
    out_flow_diff.release()
    out_dis_flow.release()
    out_ref_flow.release()
    print("Processing completed.")

# ==== 设置路径 ====
distorted_video_path = 'D:\VSR-QAD/visualization2\original_distorted_video.mp4'
reference_video_path = 'D:\VSR-QAD/fixed_hr/video002_29fps.mp4'
coarse_video_path = 'C:/D/VSR-QAD/video002/coarse_weighted_video.mp4'
fine_video_path = 'C:/D/VSR-QAD/video002/fine_weighted_video.mp4'
original_video_path = 'C:/D/VSR-QAD/video002/video093_23fps_01_x4_10.mp4'
flow_diff_video_path = 'C:/D/VSR-QAD/video002/temporalincon_video.mp4'
reference_video_output_path = 'C:/D/VSR-QAD/video002/video002_29fps.mp4'
distorted_flow_video_path = 'C:/D/VSR-QAD/video002/SR_flow_video.mp4'
reference_flow_video_path = 'C:/D/VSR-QAD/video002/reference_flow_video.mp4'

process_videos_with_filters(distorted_video_path, reference_video_path, coarse_video_path, fine_video_path,
                            original_video_path, reference_video_output_path, flow_diff_video_path,
                            distorted_flow_video_path, reference_flow_video_path)
