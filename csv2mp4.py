import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from typing import List, Tuple


# 定义固定坏行列表（基于0的索引）
BAD_ROWS = [1, 3, 4, 9, 16, 20, 24, 38]

def correct_fixed_bad_rows(matrix: np.ndarray) -> np.ndarray:
    """
    修正固定的坏行
    
    参数:
    matrix (np.ndarray): 输入矩阵
    
    返回:
    np.ndarray: 修正后的矩阵
    """
    corrected_matrix = matrix.copy()
    
    for row_idx in BAD_ROWS:
        if row_idx > 0 and row_idx < len(matrix) - 1:
            # 使用上下两行的平均值替换坏行
            corrected_matrix[row_idx] = (matrix[row_idx-1] + matrix[row_idx+1]) / 2
        elif row_idx == 0:
            # 第一行使用第二行
            corrected_matrix[row_idx] = matrix[row_idx+1]
        else:
            # 最后一行使用倒数第二行
            corrected_matrix[row_idx] = matrix[row_idx-1]
    
    return corrected_matrix

def read_csv_chunks(file_path: str, chunk_size: int = 65, correct_bad: bool = True) -> List[Tuple[str, np.ndarray]]:
    """
    从CSV文件中读取数据块，并可选地修正坏行
    
    参数:
    file_path (str): CSV文件路径
    chunk_size (int): 每个数据块的行数
    correct_bad (bool): 是否修正坏行
    
    返回:
    List[Tuple[str, np.ndarray]]: 包含时间戳和修正后矩阵的元组列表
    """
    data_chunks = []
    chunk = []
    
    with open(file_path, 'r') as f:
        for line in f:
            chunk.append(line.strip().split(','))
            if len(chunk) == chunk_size:
                timestamp = chunk[0][0]
                matrix_data = [row[1:65] for row in chunk[1:65]]
                matrix = np.array(matrix_data, dtype=float)
                
                # 修正固定坏行
                matrix = correct_fixed_bad_rows(matrix)
                
                data_chunks.append((timestamp, matrix))
                chunk = []
    
    return data_chunks

def enhance_contrast(matrix: np.ndarray, method='percentile', clip_range=(5, 95)) -> np.ndarray:
    """增强矩阵数据的对比度"""
    if method == 'percentile':
        # 使用百分位数裁剪
        vmin, vmax = np.percentile(matrix, clip_range[0]), np.percentile(matrix, clip_range[1])
        return np.clip(matrix, vmin, vmax)
    
    elif method == 'log':
        # 对数变换（适用于数据分布跨越多个数量级的情况）
        matrix_log = np.log(matrix - matrix.min() + 1)
        return matrix_log / matrix_log.max() * 255
    
    elif method == 'histogram':
        # 简单的直方图均衡化
        flat = matrix.flatten()
        hist, bins = np.histogram(flat, 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[matrix.astype('uint8')]
    
    return matrix

def create_heatmap_video(data_chunks: List[Tuple[str, np.ndarray]], output_path: str = 'heatmap_video', 
                         fps: int = 8.16, contrast_method='histogram', clip_range=(5, 95), 
                         figsize=(10, 8), dpi=300, only_heatmap=True) -> None:
    """创建热力图视频，支持对比度增强"""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 数据预处理
    processed_chunks = []
    for timestamp, matrix in data_chunks:
        processed_matrix = enhance_contrast(matrix, method=contrast_method, clip_range=clip_range)
        processed_chunks.append((timestamp, processed_matrix))
    
    # 计算颜色范围
    all_data = np.concatenate([matrix.flatten() for _, matrix in processed_chunks])
    vmin, vmax = np.percentile(all_data, 1), np.percentile(all_data, 99)
    
    # 初始化热力图
    im = ax.imshow(np.zeros((64, 64)), cmap='gray', vmin=vmin, vmax=vmax, 
                   interpolation='lanczos')
    
    # 如果只需要热力图区域，隐藏所有其他元素
    if only_heatmap:
        ax.axis('off')  # 隐藏坐标轴
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)  # 移除边距
    else:
        plt.colorbar(im)
        title = ax.set_title('')
    
    def update(frame):
        timestamp, matrix = processed_chunks[frame]
        im.set_data(matrix)
        if not only_heatmap:
            title.set_text(f'Heatmap at {timestamp}')
        return im,
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(processed_chunks), 
                                  interval=1000/fps, blit=True)
    
    # 保存视频
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()
    
    print(f"视频已保存为: {output_file}")

def main(file_path: str) -> None:
    """
    主函数：读取CSV文件并创建热力图视频
    
    参数:
    file_path (str): CSV文件路径
    """
    # 读取数据块
    data_chunks = read_csv_chunks(file_path)
    
    if not data_chunks:
        print("未找到数据块")
        return
    
    # 创建热力图视频
    video_path = 'heatmap_'+file_path[:-4]+'.mp4'
    create_heatmap_video(data_chunks, video_path)
    
    print(f"成功生成热力图视频: {video_path}")

if __name__ == "__main__":
    # 示例调用
    csv_file_path = 'video1mod.csv'  # 替换为你的CSV文件路径
    main(csv_file_path)
