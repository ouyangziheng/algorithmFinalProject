#!/usr/bin/env python3
"""
此脚本用于从视频中提取帧，并使用CLIP模型计算帧间相似度矩阵，然后将结果保存为npy文件。
生成的npy文件可以被C++程序读取和处理。
"""

import numpy as np
import os
import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import tqdm

# 导入用于视频处理和CLIP模型的库
try:
    import decord
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("警告：某些依赖库未安装，请安装：pip install decord transformers torch torchvision tqdm")
    exit(1)


def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: int = None,
) -> torch.FloatTensor:
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        width: 调整大小后的帧宽度
        height: 调整大小后的帧高度
        skip_frames_start: 从开始跳过的帧数
        skip_frames_end: 从结尾跳过的帧数
        max_num_frames: 最大采样帧数
        frame_sample_step: 采样步长
        
    Returns:
        视频帧张量 [F, H, W, C]
    """
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        start_frame = min(skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - skip_frames_end)

        if end_frame <= start_frame:
            indices = [start_frame]
        elif end_frame - start_frame <= max_num_frames:
            indices = list(range(start_frame, end_frame))
        else:
            step = frame_sample_step or (end_frame - start_frame) // max_num_frames
            indices = list(range(start_frame, end_frame, step))

        frames = video_reader.get_batch(indices=indices)
        frames = frames[:max_num_frames].float()  # 确保不超过限制

        # 正规化帧
        transform = T.Lambda(lambda x: x / 255.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames  # [F, H, W, C]


def compute_clip_similarity(frames: torch.Tensor, device: str) -> torch.Tensor:
    """
    计算视频帧之间的CLIP相似度矩阵
    
    Args:
        frames: 视频帧 [F, H, W, C]
        device: 计算设备 ('cuda' 或 'cpu')
        
    Returns:
        相似度矩阵 [F, F]
    """
    # 加载CLIP模型
    print("加载CLIP模型...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    num_frames = frames.shape[0]
    similarity_matrix = torch.zeros((num_frames, num_frames), device=device)
    
    # 将帧转换为PIL图像列表
    print("处理视频帧...")
    pil_frames = []
    for i in range(num_frames):
        # 转换为PIL图像
        pil_image = Image.fromarray((frames[i].cpu().numpy() * 255).astype(np.uint8))
        pil_frames.append(pil_image)
    
    # 处理所有图像以获取特征
    print(f"计算CLIP特征 (共{num_frames}帧)...")
    inputs = processor(images=pil_frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # 归一化特征
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    
    # 计算相似度矩阵
    print("计算相似度矩阵...")
    similarity_matrix = torch.mm(image_features, image_features.t())
    
    return similarity_matrix


def save_similarity_matrix(similarity_matrix: torch.Tensor, output_path: str, prefix: str = "similarity"):
    """
    保存相似度矩阵为npy文件
    
    Args:
        similarity_matrix: 相似度矩阵 [F, F]
        output_path: 输出目录
        prefix: 输出文件前缀
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(similarity_matrix, torch.Tensor):
        sim_matrix_np = similarity_matrix.cpu().numpy()
    else:
        sim_matrix_np = similarity_matrix
    
    # 保存为npy文件
    np_output_file = os.path.join(output_path, f"{prefix}_matrix.npy")
    np.save(np_output_file, sim_matrix_np)
    print(f"相似度矩阵数据已保存至: {np_output_file}")


def main():
    parser = argparse.ArgumentParser(description="从视频中提取帧并计算CLIP相似度矩阵")
    
    # 视频处理参数
    parser.add_argument("--video_path", type=str, required=True, help="视频文件路径")
    parser.add_argument("--output_path", type=str, default="output", help="输出结果的路径")
    parser.add_argument("--width", type=int, default=720, help="调整大小后的视频帧宽度")
    parser.add_argument("--height", type=int, default=480, help="调整大小后的视频帧高度")
    parser.add_argument("--skip_frames_start", type=int, default=0, help="从开始跳过的帧数")
    parser.add_argument("--skip_frames_end", type=int, default=0, help="从结尾跳过的帧数")
    parser.add_argument("--frame_sample_step", type=int, default=None, help="采样帧的时间步长")
    parser.add_argument("--max_num_frames", type=int, default=101, help="最大采样帧数")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--prefix", type=str, default="similarity", help="输出文件前缀")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"使用设备: {device}")
    
    # 提取视频帧
    print(f"正在读取视频: {args.video_path}")
    frames = get_video_frames(
        video_path=args.video_path,
        width=args.width,
        height=args.height,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        max_num_frames=args.max_num_frames,
        frame_sample_step=args.frame_sample_step,
    )
    
    print(f"提取的视频帧数: {frames.shape[0]}")
    
    # 计算CLIP相似度
    similarity_matrix = compute_clip_similarity(frames, device)
    
    # 保存相似度矩阵
    save_similarity_matrix(similarity_matrix, args.output_path, args.prefix)
    
    print("处理完成!")


if __name__ == "__main__":
    main() 