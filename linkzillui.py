
import struct
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QWidget, QTextEdit, QVBoxLayout, QApplication, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QProgressBar
import sys, time, csv, socket, os
from datetime import datetime
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import zoom

dark_cur = None
light_cur = None

def Load_Csv(csv_file_path):
    return np.loadtxt(csv_file_path, delimiter=',')

def Save_Csv(fileName, data):
    csv_file_path = fileName + '.csv'
    np.savetxt(csv_file_path, data, delimiter=',', fmt='%.8f')

class HexTCPClient(QThread):
    update_img = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    """支持持久连接的TCP客户端，用于发送和接收十六进制数据"""
    def __init__(self, ui, port=3192, timeout=5):
        """
        初始化TCP客户端
        
        Args:
            host: 服务器主机名
            port: 服务器端口
            timeout: 连接超时时间(秒)
        """
        QThread.__init__(self)
        self.ui = ui
        self.host = "192.168." + str(self.ui.ip_3.value()) + '.'+ str(self.ui.ip_4.value())
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.is_connected = False
        self.img = np.zeros((64, 64), dtype=float)
        self.is_scanning = False
        self.save_dark = False
        self.save_light = False
        self.keep_scan = False
        self.scan_once = False
        self.path = None
    
    def run(self):
        """连接到服务器"""
        try:
            # 创建TCP套接字
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            # 连接到服务器
            print(f"连接到 {self.host}:{self.port}...")
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            print("连接成功")

        except socket.timeout:
            print("连接超时")
        except ConnectionRefusedError:
            print(f"连接被拒绝，请检查服务器是否运行在 {self.host}:{self.port}")
        except Exception as e:
            print(f"连接错误: {e}")
        while self.is_connected:
            if self.scan_once:
                self.scan(self.path)
                self.scan_once = False
            if self.keep_scan:
                self.multi_scan(self.path)
                self.keep_scan = False


    
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
                
                #print(f"等待接收 {expected_size} 字节数据...")
                
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
                        '''
                        # 打印进度
                        if len(chunk) > 0:
                            print(f"已接收: {len(received_data)}/{expected_size} 字节")
                        '''
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
            if int_list[0] == 18 and int_list[1] == 52 and int_list[-1] == 33 and int_list[-2] == 67: # 校验 12 34 ... 43 21 hex
                return int_list[2:-2]
            else:
                print("readout error 1")
        return None

    def convert_to_hex(self, f1, f2, f3, f4, i):
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
        hex_f1 = ''.join([hex_f1[i:i + 2] for i in range(0, len(hex_f1), 2)][::-1])

        hex_f2 = struct.pack('>f', f2).hex()
        hex_f2 = ''.join([hex_f2[i:i + 2] for i in range(0, len(hex_f2), 2)][::-1])

        hex_f3 = struct.pack('>f', f3).hex()
        hex_f3 = ''.join([hex_f3[i:i + 2] for i in range(0, len(hex_f3), 2)][::-1])

        hex_f4 = struct.pack('>f', f4).hex()
        hex_f4 = ''.join([hex_f4[i:i + 2] for i in range(0, len(hex_f4), 2)][::-1])

        hex_i = struct.pack('>B', i).hex()

        # 拼接所有十六进制字符串
        result = 'FF00' + hex_f1 + hex_f2 + hex_f3 + hex_f4 + hex_i + '0D0A'
        return result

    def set_range(self):
        self.range = self.ui.Range.value()
        self.send_hex_string(self.convert_to_hex(self.ui.V1.value(),
                                                 self.ui.V2.value(),
                                                 self.ui.Von.value(),
                                                 self.ui.Voff.value(),
                                                 self.range))

    def matricization(self, a):

        factors = np.array([3355.4432, 13.1072, 0.0512, 0.0002])
        size = self.img[0].size
        coefficient = self.range / 200

        # 填充矩阵
        for i in range(size):
            for j in range(size - 1, -1, -1):
                k = i * size + (size - 1 - j)
                # 计算加权和
                weighted_sum = np.sum(a[k * 4:k * 4 + 4] * factors)
                # 应用系数
                self.img[i][j] = weighted_sum * coefficient

    def scan(self, filename):
        self.is_scanning = True
        self.matricization(self.send_and_receive("FF01 010D 0A"))
        Save_Csv(filename, self.img)
        self.update_img.emit(self.img)
        self.is_scanning = False

    def multi_scan(self, folder_path):
        start_point = datetime.now()
        while self.keep_scan and not self.is_scanning:
            time_diff = datetime.now() - start_point
            frame_st = datetime.now()
            filename = folder_path + "/RawData_" + str(time_diff.total_seconds())
            self.scan(filename)
            frame_time = datetime.now() - frame_st
            fps = 1 / frame_time.total_seconds()
            self.fps_updated.emit(fps)
            if self.save_dark:
                global dark_cur
                dark_cur = np.copy(self.img)
                Save_Csv(folder_path + "/Dark", self.img)
                self.save_dark = False
            if self.save_light:
                global light_cur
                light_cur = np.copy(self.img)
                Save_Csv(folder_path + "/Light", self.img)
                self.save_light = False
            time.sleep(0.1)

    def __del__(self):
        self.disconnect()


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class Stats(QMainWindow):

    def __init__(self):
        super().__init__()
        # Load UI
        self.ui = uic.loadUi("ScanGUI.ui", self)
        self.setWindowTitle("ScanGUI")

        # Output Display
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.outputTextEdit = self.ui.findChild(QTextEdit, "Console")
        self.img1 = np.random.randn(64, 64)
        # Parameters

        # Events
        self.ui.conn_init.clicked.connect(self.Connect_Init)
        self.ui.setting.clicked.connect(self.Setting)
        self.ui.photo.clicked.connect(self.Shoot_Picture)
        self.ui.scan.toggled.connect(self.Video_Mode)
        self.ui.save_dark.clicked.connect(self.Save_Dark)
        self.ui.save_light.clicked.connect(self.Save_Light)
        # Form Greyscale Color Map
        colors = [(i, i, i) for i in range(256)]
        self.colormap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=colors)

        # Figures Initialization
        self.plot1 = self.ui.IMG1.addPlot()
        self.img_item1 = pg.ImageItem()
        self.plot1.addItem(self.img_item1)
        self.img_item1.setLookupTable(self.colormap.getLookupTable())
        self.plot1.setAspectLocked(True)
        self.plot1.setMouseEnabled(x=False, y=False)
        self.plot1.hideAxis('bottom')
        self.plot1.hideAxis('left')
        self.img_item1.setImage(self.img1)
        self.img_item2 = pg.ImageItem()
        self.plot2 = self.ui.IMG2.addPlot()
        self.plot2.addItem(self.img_item2)
        self.img_item2.setLookupTable(self.colormap.getLookupTable())
        self.img_item2.setImage(self.img1)
        self.show()

        self.scene = QGraphicsScene()

        self.IPT = ImageProcessThread(self.ui)
        self.IPT.start()
        self.IPT.updater.connect(self.Updated_Enhanced_Image)

    @pyqtSlot(str)
    def normalOutputWritten(self, text):
        cursor = self.outputTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.outputTextEdit.setTextCursor(cursor)
        self.outputTextEdit.ensureCursorVisible()

    def Connect_Init(self):
        self.TCP = HexTCPClient(self.ui)
        self.TCP.start()
        self.TCP.update_img.connect(self.Update_Img1)
        self.TCP.fps_updated.connect(self.Update_FPS)

    def Setting(self):
        try:
            # 连接到服务器
            if not self.TCP.is_connected:
                print('TCP error')
            self.TCP.set_range()
        except:
            print("Setting Error")

    def Shoot_Picture(self):
        folder_path = "Data/" + self.ui.File_Name.text()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = folder_path +"/PicRawData_"+str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not self.TCP.is_scanning:
            self.TCP.path = filename
            self.TCP.scan_once = True

    def Video_Mode(self):
        if self.ui.scan.isChecked():
            self.ui.save_light.setEnabled(True)
            self.ui.save_dark.setEnabled(True)
            self.ui.setting.setEnabled(False)
            self.ui.photo.setEnabled(False)
            self.folder_path = "Data/"+ self.ui.File_Name.text() +"/Video_" + str(datetime.now().strftime("%Y%m%d_%H%M%S"))
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
            self.TCP.path = self.folder_path
            self.TCP.keep_scan = True
        else:
            self.TCP.keep_scan = False
            self.ui.save_light.setEnabled(False)
            self.ui.save_dark.setEnabled(False)
            self.ui.photo.setEnabled(True)
            self.ui.setting.setEnabled(True)

    def Save_Dark(self):
        self.TCP.save_dark = True

    def Save_Light(self):
        self.TCP.save_light = True

    @pyqtSlot(np.ndarray)
    def Update_Img1(self, img):
        if self.ui.scan.isChecked() and not self.IPT.is_busy and (dark_cur is not None) and (light_cur is not None):
            self.IPT.raw_data.emit(img)
        self.img_item1.setImage(img)

    @pyqtSlot(np.ndarray)
    def Updated_Enhanced_Image(self, img):
        self.img_item2.setImage(img)
        Save_Csv(self.folder_path + "/Enhanced" + str(datetime.now().strftime("%Y%m%d_%H%M%S")), img)
    @pyqtSlot(float)
    def Update_FPS(self, fps):
        self.ui.fps.setText("FPS: " + str(fps))

class ImageProcessThread(QThread):

    updater = pyqtSignal(np.ndarray)
    raw_data = pyqtSignal(np.ndarray)
    def __init__(self, ui):
        QThread.__init__(self)
        self.ui = ui
        self.is_running = False
        self.is_busy = False
        self.raw_data.connect(self.trigger)

    def run(self):
        self.is_running = True
        self.new_task = False
        while self.is_running:
            if self.new_task:
                #self.normalize(self.raw)
                self.lanczos(self.raw)
                self.new_task = False
            time.sleep(0.01)

    @pyqtSlot(np.ndarray)
    def trigger(self, matrix):
        self.new_task = True
        self.raw = matrix

    def lanczos(self, matrix):
        self.is_busy = True
        zoomed_image = zoom(matrix, zoom=10, order=4)
        self.updater.emit(zoomed_image)
        self.is_busy = False

    def normalize(self, matrix):
        self.is_busy = True
        global light_cur, dark_cur
        denominator = light_cur - dark_cur
        denominator = np.maximum(denominator, 0.0001)
        normalized = (matrix - dark_cur) / denominator
        normalized = np.clip(normalized, 0, 1)
        self.updater.emit(normalized)
        self.is_busy = False

    def stop(self):
        self.is_running = False




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    stats = Stats()
    stats.show()
    app.exec_()

