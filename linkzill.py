
import struct
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QWidget, QTextEdit, QVBoxLayout, QApplication, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QProgressBar
from pyqtgraph.exporters import ImageExporter
import sys, time, csv, socket
from datetime import datetime
import numpy as np

def convert_to_hex(f1, f2, f3, f4, i):
    """
    将4个浮点数和1个整数转换为特定格式的十六进制字符串
    
    参数:
    f1, f2, f3, f4: 四个浮点数
    i: 一个short整数
    
    返回:
    拼接后的十六进制字符串（大写）
    """
    # 转换浮点数并倒置字节顺序，然后转为大写
    hex_f1 = struct.pack('>f', f1).hex()
    hex_f1 = ''.join([hex_f1[i:i+2] for i in range(0, len(hex_f1), 2)][::-1])
    
    hex_f2 = struct.pack('>f', f2).hex()
    hex_f2 = ''.join([hex_f2[i:i+2] for i in range(0, len(hex_f2), 2)][::-1])
    
    hex_f3 = struct.pack('>f', f3).hex()
    hex_f3 = ''.join([hex_f3[i:i+2] for i in range(0, len(hex_f3), 2)][::-1])
    
    hex_f4 = struct.pack('>f', f4).hex()
    hex_f4 = ''.join([hex_f4[i:i+2] for i in range(0, len(hex_f4), 2)][::-1])
    
    hex_i = struct.pack('>B', i).hex()
    
    # 拼接所有十六进制字符串
    result = 'FF00' + hex_f1 + hex_f2 + hex_f3 + hex_f4 + hex_i + '0D0A'
    return result

def matrix_calculation(a, cur_level, size=64):
    # 初始化结果矩阵
    b = np.zeros((size, size), dtype=float)
    
    # 计算因子
    factors = np.array([3355.4432, 13.1072, 0.0512, 0.0002])
    
    # 计算系数
    coefficient = cur_level / 200
    
    # 填充矩阵
    for i in range(size):
        for j in range(size-1, -1, -1):
            k = i * size + (size - 1 - j)
            # 计算加权和
            weighted_sum = np.sum(a[k*4:k*4+4] * factors)
            # 应用系数
            b[size-j-1][i] = weighted_sum * coefficient
    return b

class HexTCPClient:
    """支持持久连接的TCP客户端，用于发送和接收十六进制数据"""
    
    def __init__(self, host='localhost', port=3192, timeout=5):
        """
        初始化TCP客户端
        
        Args:
            host: 服务器主机名
            port: 服务器端口
            timeout: 连接超时时间(秒)
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.is_connected = False
    
    def connect(self):
        """连接到服务器"""
        if self.is_connected:
            print("已经连接到服务器")
            return True
        
        try:
            # 创建TCP套接字
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            # 连接到服务器
            print(f"连接到 {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            print("连接成功")
            return True
            
        except socket.timeout:
            print("连接超时")
            return False
        except ConnectionRefusedError:
            print(f"连接被拒绝，请检查服务器是否运行在 {self.host}:{self.port}")
            return False
        except Exception as e:
            print(f"连接错误: {e}")
            return False
    
    def disconnect(self):
        """断开与服务器的连接"""
        if not self.is_connected:
            return
        
        try:
            self.socket.close()
        except:
            pass
        
        self.is_connected = False
        print("已断开连接")
    
    def send_hex_string(self, hex_string):
        """
        以字符串形式发送十六进制数据
        
        Args:
            hex_string: 十六进制字符串，例如 "FF01010D0A"
            
        Returns:
            是否发送成功
        """
        if not self.is_connected:
            print("未连接到服务器")
            return False
        
        try:
            # 将十六进制字符串转换为字节
            bytes_data = bytes.fromhex(hex_string)
            
            # 发送数据
            print(f"发送十六进制字符串: {hex_string}")
            self.socket.sendall(bytes_data)
            return True
            
        except Exception as e:
            print(f"发送错误: {e}")
            self.is_connected = False
            return False
    
    def receive_hex_string(self, expected_size=None, timeout=1.0, buffer_size=16388):
        """
        可靠地接收数据并以十六进制字符串形式返回
        
        Args:
            expected_size: 期望接收的字节数，如果为None则只接收一次
            timeout: 接收超时时间(秒)
            buffer_size: 单次接收缓冲区大小
            
        Returns:
            接收到的十六进制字符串，失败时返回None
        """
        if not self.is_connected:
            print("未连接到服务器")
            return None
        
        try:
            # 设置超时
            original_timeout = self.socket.gettimeout()
            self.socket.settimeout(timeout)
            
            # 接收数据
            received_data = b''
            
            if expected_size is not None:
                # 接收指定大小的数据
                remaining = expected_size
                start_time = time.time()
                
                print(f"等待接收 {expected_size} 字节数据...")
                
                while remaining > 0:
                    # 检查超时
                    if time.time() - start_time > timeout:
                        print(f"接收超时，仅收到 {len(received_data)}/{expected_size} 字节")
                        break
                    
                    try:
                        # 接收数据块
                        chunk = self.socket.recv(min(remaining, buffer_size))
                        
                        # 检查连接是否关闭
                        if not chunk:
                            print("连接已关闭，未接收完所有数据")
                            self.is_connected = False
                            break
                        
                        received_data += chunk
                        remaining -= len(chunk)
                        
                        # 打印进度
                        if len(chunk) > 0:
                            print(f"已接收: {len(received_data)}/{expected_size} 字节")
                        
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"接收过程中出错: {e}")
                        self.is_connected = False
                        break
            else:
                # 只接收一次数据（兼容旧行为）
                print("等待响应...")
                received_data = self.socket.recv(buffer_size)
            
            # 恢复原始超时设置
            self.socket.settimeout(original_timeout)
            
            # 转换为十六进制字符串
            if received_data:
                hex_response = ''.join(f'{b:02X}' for b in received_data)
                print(f"成功接收: {len(received_data)} 字节，十六进制长度: {len(hex_response)}")
                return hex_response
            else:
                print("没有接收到响应，连接可能已关闭")
                self.is_connected = False
                return None
                    
        except Exception as e:
            print(f"接收错误: {e}")
            self.is_connected = False
            return None
    
    def send_and_receive(self, hex_string, buffer_size=16388):
        if self.send_hex_string(hex_string):
            hex_string_r = self.receive_hex_string(buffer_size)
            int_list = [int(hex_string_r[i:i+2], 16) for i in range(0, len(hex_string_r), 2)]
            if int_list[0] == 18 and int_list[1] == 52 and int_list[-1] == 33 and int_list[-2] == 67: # 校验 12 34 ... 43 21
                print("update")
                return int_list[2:-2]
            else:
                print("readout error 1")
        return None
class MatrixHeatmapUpdater:
    """极简64×64矩阵热力图更新器"""
    
    def __init__(self, initial_data=None, title="矩阵热力图", cmap='gray', figsize=(10, 8)):
        """
        初始化热力图更新器
        
        参数:
            initial_data: 初始64×64矩阵数据，默认为随机矩阵
            title: 热力图标题
            cmap: 颜色映射方案
            figsize: 图形大小
        """
        # 设置初始数据
        self.data = initial_data if initial_data is not None else np.random.rand(64, 64)
        
        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # 创建热力图
        self.heatmap = self.ax.imshow(
            self.data, 
            cmap=cmap, 
            interpolation='nearest'
        )
        
        # 添加颜色条
        self.fig.colorbar(self.heatmap, ax=self.ax)
        
        # 设置标题
        self.ax.set_title(title)
        
        # 隐藏坐标轴
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # 调整布局
        plt.tight_layout()
    
    def update(self, new_data):
        """
        更新热力图数据
        
        参数:
            new_data: 新的64×64矩阵数据
        """
        # 更新数据
        self.data = new_data
        
        # 更新热力图
        self.heatmap.set_data(self.data)
        
        # 自动调整颜色范围
        self.heatmap.set_clim(np.min(self.data), np.max(self.data))
        
        # 重绘
        self.fig.canvas.draw_idle()
        
        # 显示更新后的图形
        plt.pause(0.001)
    
    def show(self):
        """显示热力图窗口"""
        plt.show()

class HeatmapInterpolator:
    """
    支持多种插值方法的热力图更新器
    
    支持的插值方法:
    - 'nearest': 最近邻插值
    - 'bilinear': 双线性插值
    - 'bicubic': 双三次插值
    - 'lanczos': Lanczos插值（高质量）
    """
    
    def __init__(self, matrix_size=64, cmap='gray', interpolation='lanczos', figsize=(8, 8)):
        """
        初始化热力图更新器
        
        参数:
            matrix_size: 矩阵大小
            cmap: 颜色映射方案
            interpolation: 插值方法
            figsize: 图形大小
        """
        self.matrix_size = matrix_size
        self.interpolation = interpolation
        
        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # 初始化数据
        self.data = np.zeros((matrix_size, matrix_size))
        
        # 创建热力图
        self.heatmap = self.ax.imshow(
            self.data, 
            cmap=cmap, 
            interpolation=interpolation,
            vmin=0, 
            vmax=1
        )
        
        # 添加颜色条
        self.fig.colorbar(self.heatmap)
        
        # 设置标题
        self.ax.set_title(f'热力图 ({interpolation}插值)')
        
        # 开启交互模式
        plt.ion()
        
        # 显示图形
        plt.show()
    
    def update(self, new_data, vmin=None, vmax=None):
        """
        更新热力图数据
        
        参数:
            new_data: 新的矩阵数据
            vmin, vmax: 颜色映射的最小值和最大值，None表示自动调整
        """
        # 确保数据形状正确
        if new_data.shape != (self.matrix_size, self.matrix_size):
            new_data = np.resize(new_data, (self.matrix_size, self.matrix_size))
        
        # 更新数据
        self.data = new_data
        
        # 更新热力图
        self.heatmap.set_data(self.data)
        
        # 更新颜色范围
        if vmin is not None and vmax is not None:
            self.heatmap.set_clim(vmin, vmax)
        else:
            self.heatmap.set_clim(np.min(self.data), np.max(self.data))
        
        # 重绘
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
    
    def set_interpolation(self, method):
        """
        设置插值方法
        
        参数:
            method: 插值方法名称
        """
        valid_methods = ['nearest', 'bilinear', 'bicubic', 'lanczos']
        if method in valid_methods:
            self.interpolation = method
            self.heatmap.set_interpolation(method)
            self.ax.set_title(f'热力图 ({method}插值)')
            print(f"插值方法已设置为: {method}")
        else:
            print(f"不支持的插值方法，可用: {valid_methods}")
    
    def close(self):
        """关闭图形窗口"""
        plt.ioff()
        plt.close(self.fig)

def detect_bad_points(matrix, threshold=10.0):
    """
    检测矩阵中超过阈值的坏点
    
    参数:
        matrix: 输入的二维数组
        threshold: 坏点阈值，默认10.0
        
    返回:
        bad_points: 坏点坐标列表 [(row1, col1), (row2, col2), ...]
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # 创建掩码矩阵，标记超过阈值的点
    bad_mask = matrix > threshold
    
    # 获取坏点坐标
    bad_points = [(i, j) for i in range(matrix.shape[0]) 
                 for j in range(matrix.shape[1]) if bad_mask[i, j]]
    
    return bad_points


def correct_bad_points(bad_points, matrix, window_size=3, method='mean'):
    """
    使用查表法对指定坏点进行插值校正
    
    参数:
        bad_points: 坏点坐标列表，格式为 [(row1, col1), (row2, col2), ...]
        matrix: 输入矩阵
        window_size: 邻域窗口大小，默认为3
        method: 插值方法，可选 'mean'(均值), 'median'(中值)
    
    返回:
        corrected: 校正后的矩阵
    """
    rows, cols = matrix.shape
    corrected = matrix.copy()
    
    # 创建坏点掩码（查表法核心）
    is_bad = np.zeros((rows, cols), dtype=bool)
    for i, j in bad_points:
        if 0 <= i < rows and 0 <= j < cols:
            is_bad[i, j] = True
    
    print(f"标记了 {len(bad_points)} 个坏点")
    
    # 邻域窗口半径
    radius = window_size // 2
    
    # 对每个坏点进行校正
    for i, j in bad_points:
        if 0 <= i < rows and 0 <= j < cols:
            # 收集邻域内的有效点
            valid_neighbors = []
            
            # 遍历邻域窗口
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    # 跳过自身
                    if di == 0 and dj == 0:
                        continue
                        
                    ni, nj = i + di, j + dj
                    
                    # 检查坐标有效性
                    if 0 <= ni < rows and 0 <= nj < cols:
                        # 通过查表快速判断是否为坏点
                        if not is_bad[ni, nj]:
                            valid_neighbors.append(matrix[ni, nj])
            
            # 根据插值方法计算校正值
            if valid_neighbors:
                if method == 'mean':
                    corrected[i, j] = np.mean(valid_neighbors)
                elif method == 'median':
                    corrected[i, j] = np.median(valid_neighbors)
            # 若无有效邻域点，则保持原值不变
    
    return corrected


if __name__ == "__main__":
    SERVER_HOST = '192.168.137.110'
    SERVER_PORT = 3192
    
    # 创建客户端实例
    client = HexTCPClient(SERVER_HOST, SERVER_PORT)

    #updater = MatrixHeatmapUpdater(title="实时更新热力图")
    updater = HeatmapInterpolator(
        matrix_size=64,           # 矩阵大小
        interpolation='lanczos',  # 插值方法
        figsize=(10, 8)           # 图形大小
    )

    try:
        # 连接到服务器
        if not client.connect():
            print('error')
        client.send_hex_string(convert_to_hex(0.1,0.1,10,-10,1))
        time.sleep(1)
        response = client.send_and_receive("FF01 010D 0A")
        while True:
            response = client.send_and_receive("FF01 010D 0A")
            if response == None:
                continue
            image = matrix_calculation(response, 20)
            bad_points = [(2, 2), (2, 48), (6, 32), (7, 54), (20, 50), (21, 48), (21, 49), (30, 60), (31, 2), (35, 42), (46, 28)]
            #bad_points = detect_bad_points(image, threshold=5.0)
            #print(bad_points)
            correct_img = correct_bad_points(bad_points, image)

            updater.update(correct_img)

    finally:
        # 确保断开连接
        client.disconnect()
        plt.ioff()  # 关闭交互模式
        plt.show()
        app.exec_()
