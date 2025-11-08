import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from datetime import datetime
from yolo_detector import YOLODetector

class YOLOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 目标检测工具")
        self.root.geometry("600x500")
        
        # 初始化结果输出文件
        self.init_result_file()
        
        self.model_path = tk.StringVar()
        self.source_path = tk.StringVar()
        self.conf_threshold = tk.DoubleVar(value=0.30)
        self.detector = None
        self.is_detecting = False
        
        self.create_widgets()
        
    def init_result_file(self):
        """初始化结果输出文件"""
        try:
            with open("out.txt", "w", encoding="utf-8") as f:
                f.write("YOLOv8 目标检测结果\n")
                f.write(f"检测开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"初始化结果文件失败: {e}")
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 模型选择
        model_frame = ttk.LabelFrame(main_frame, text="模型设置", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="模型文件:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=(5, 5))
        ttk.Button(model_frame, text="浏览", command=self.browse_model).grid(row=0, column=2)
        
        # 输入源选择
        source_frame = ttk.LabelFrame(main_frame, text="输入源设置", padding="10")
        source_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(source_frame, text="输入文件:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(source_frame, textvariable=self.source_path, width=50).grid(row=0, column=1, padx=(5, 5))
        ttk.Button(source_frame, text="浏览", command=self.browse_source).grid(row=0, column=2)
        
        # 摄像头选项
        ttk.Button(source_frame, text="使用摄像头", command=self.use_camera).grid(row=1, column=1, pady=(5, 0))
        
        # 参数设置
        param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding="10")
        param_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(param_frame, text="置信度阈值:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(param_frame, from_=0.01, to=1.0, variable=self.conf_threshold, orient=tk.HORIZONTAL, length=300).grid(row=0, column=1, padx=(5, 5))
        self.conf_label = ttk.Label(param_frame, text=f"{self.conf_threshold.get():.2f}")
        self.conf_label.grid(row=0, column=2)
        
        # 更新置信度标签
        def update_conf_label(val):
            self.conf_label.config(text=f"{float(val):.2f}")
        scale = param_frame.winfo_children()[1]  # 获取Scale控件
        scale.configure(command=update_conf_label)
        
        # 控制按钮
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        self.start_button = ttk.Button(control_frame, text="开始检测", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.exit_button = ttk.Button(control_frame, text="退出", command=self.root.quit)
        self.exit_button.grid(row=0, column=2)
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 日志文本框
        log_frame = ttk.LabelFrame(main_frame, text="日志信息", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.log_text = tk.Text(log_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型文件", "*.pt"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path.set(filename)
            
    def browse_source(self):
        filename = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[
                ("图片和视频文件", "*.png *.jpg *.jpeg *.mp4 *.avi *.mov"),
                ("图片文件", "*.png *.jpg *.jpeg"),
                ("视频文件", "*.mp4 *.avi *.mov"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.source_path.set(filename)
            
    def use_camera(self):
        self.source_path.set("camera")
        
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_detection(self):
        # 检查必要参数
        if not self.model_path.get():
            messagebox.showerror("错误", "请选择模型文件")
            return
            
        if not self.source_path.get():
            messagebox.showerror("错误", "请选择输入源或使用摄像头")
            return
            
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path.get()) and self.model_path.get() != "":
            messagebox.showerror("错误", "模型文件不存在")
            return
            
        # 禁用开始按钮，启用停止按钮
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.is_detecting = True
        
        # 在新线程中运行检测
        detection_thread = threading.Thread(target=self.run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
    def stop_detection(self):
        self.is_detecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        self.log_message("检测已停止")
        
    def run_detection(self):
        try:
            # 初始化检测器
            if self.detector is None or self.detector.model_path != self.model_path.get():
                self.log_message("正在加载模型...")
                self.detector = YOLODetector(self.model_path.get())
                self.log_message("模型加载完成")
                
            source = self.source_path.get()
            conf = self.conf_threshold.get()
            
            self.log_message(f"开始检测: {source}")
            self.log_message(f"置信度阈值: {conf}")
            
            # 记录检测开始
            self.record_detection_start(source, conf)
            
            if source == "camera":
                # 摄像头检测
                self.log_message("启动摄像头检测，按 'q' 键退出检测窗口")
                self.detector.detect_camera(conf_threshold=conf)
                # 记录摄像头检测结果
                self.record_detection_end("camera", "completed")
            elif os.path.isfile(source):
                # 文件检测
                if source.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 图片检测
                    self.log_message("正在进行图片检测...")
                    self.detector.detect_image(source, conf_threshold=conf)
                    self.log_message("图片检测完成")
                    # 记录图片检测结果
                    self.record_detection_end("image", "completed")
                elif source.lower().endswith(('.mp4', '.avi', '.mov')):
                    # 视频检测
                    self.log_message("正在进行视频检测...")
                    self.detector.detect_video(source, conf_threshold=conf)
                    self.log_message("视频检测完成")
                    # 记录视频检测结果
                    self.record_detection_end("video", "completed")
                else:
                    self.log_message("不支持的文件格式")
                    self.record_detection_end("unknown", "unsupported_format")
            else:
                self.log_message("输入源不存在")
                self.record_detection_end("unknown", "source_not_exist")
                
        except Exception as e:
            self.log_message(f"检测出错: {str(e)}")
            self.record_detection_end("unknown", f"error: {str(e)}")
        finally:
            if self.is_detecting:  # 只有在没有被手动停止的情况下才重置按钮状态
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.progress.stop()
            self.log_message("检测任务结束")
    
    def record_detection_start(self, source, conf):
        """记录检测开始信息"""
        try:
            with open("out.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始检测\n")
                f.write(f"输入源: {source}\n")
                f.write(f"置信度阈值: {conf}\n")
        except Exception as e:
            print(f"记录检测开始信息失败: {e}")
    
    def record_detection_end(self, detection_type, status):
        """记录检测结束信息"""
        try:
            with open("out.txt", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检测结束\n")
                f.write(f"检测类型: {detection_type}\n")
                f.write(f"状态: {status}\n")
                f.write("-" * 30 + "\n")
        except Exception as e:
            print(f"记录检测结束信息失败: {e}")

def main():
    root = tk.Tk()
    app = YOLOGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()