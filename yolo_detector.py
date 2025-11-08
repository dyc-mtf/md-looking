import cv2
import torch
from ultralytics import YOLO
import argparse
import os
import numpy as np
from datetime import datetime

class YOLODetector:
    def __init__(self, model_path):
        """
        初始化YOLO检测器
        
        Args:
            model_path (str): 模型文件路径
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        
    def detect_image(self, image_path, conf_threshold=0.30, save_result=True):
        """
        对单张图片进行目标检测
        
        Args:
            image_path (str): 图片路径
            conf_threshold (float): 置信度阈值
            save_result (bool): 是否保存结果图片
            
        Returns:
            results: 检测结果
        """
        # 加载图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        # 执行检测
        results = self.model(img, conf=conf_threshold)
        
        # 获取检测结果
        result = results[0]
        
        # 在图片上绘制检测框
        annotated_img = result.plot()
        
        # 保存结果
        if save_result:
            output_path = f"result_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_img)
            print(f"检测结果已保存到: {output_path}")
            
        # 将检测结果写入out.txt
        self._write_results_to_file(result, image_path, "image")
            
        return results
        
    def detect_video(self, video_path, conf_threshold=0.30, save_result=True):
        """
        对视频进行目标检测
        
        Args:
            video_path (str): 视频路径
            conf_threshold (float): 置信度阈值
            save_result (bool): 是否保存结果视频
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入对象
        if save_result:
            output_path = f"result_{os.path.basename(video_path)}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 执行检测
            results = self.model(frame, conf=conf_threshold)
            result = results[0]
            
            # 绘制检测结果
            annotated_frame = result.plot()
            
            # 显示实时检测结果
            cv2.imshow('YOLOv8 Detection', annotated_frame)
            
            # 保存帧
            if save_result:
                out.write(annotated_frame)
                
            # 累计检测结果
            if result.boxes is not None:
                total_detections += len(result.boxes)
                
            frame_count += 1
            print(f"已处理 {frame_count} 帧", end='\r')
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # 释放资源
        cap.release()
        if save_result:
            out.release()
        cv2.destroyAllWindows()
        
        if save_result:
            print(f"\n检测结果视频已保存到: result_{os.path.basename(video_path)}")
            
        # 将检测结果写入out.txt
        summary = f"视频 {video_path} 总共处理了 {frame_count} 帧，检测到 {total_detections} 个目标"
        self._write_results_to_file(None, video_path, "video", summary)
            
    def _write_results_to_file(self, result, source_path, detection_type, summary=None):
        """
        将检测结果写入out.txt文件
        
        Args:
            result: 检测结果对象
            source_path (str): 源文件路径
            detection_type (str): 检测类型 (image/video/camera)
            summary (str): 概要信息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open("out.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"检测时间: {timestamp}\n")
            f.write(f"检测类型: {detection_type}\n")
            f.write(f"源文件: {source_path}\n")
            
            if summary:
                f.write(f"检测摘要: {summary}\n")
            
            if result is not None and result.boxes is not None:
                # 获取类别名称
                names = result.names if hasattr(result, 'names') else {}
                
                f.write(f"检测到的目标数量: {len(result.boxes)}\n")
                f.write("详细检测结果:\n")
                
                for i, box in enumerate(result.boxes):
                    # 获取边界框坐标
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls = int(box.cls[0].cpu().item())
                    
                    # 获取类别名称
                    class_name = names.get(cls, f"类别 {cls}")
                    
                    f.write(f"  目标 {i+1}: {class_name} (置信度: {conf:.2f}) "
                           f"位置: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]\n")
            else:
                f.write("未检测到目标\n")
            
            f.write(f"{'='*50}\n")
            
    def detect_camera(self, camera_index=0, conf_threshold=0.30):
        """
        使用摄像头进行实时目标检测
        
        Args:
            camera_index (int): 摄像头索引
            conf_threshold (float): 置信度阈值
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            raise ValueError("无法打开摄像头")
            
        print("按 'q' 键退出摄像头检测模式")
        
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 执行检测
            results = self.model(frame, conf=conf_threshold)
            result = results[0]
            
            # 绘制检测结果
            annotated_frame = result.plot()
            
            # 显示结果
            cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)
            
            # 累计检测结果
            if result.boxes is not None:
                total_detections += len(result.boxes)
            
            frame_count += 1
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        
        # 将检测结果写入out.txt
        summary = f"摄像头检测总共处理了 {frame_count} 帧，检测到 {total_detections} 个目标"
        self._write_results_to_file(None, f"camera_{camera_index}", "camera", summary)

def main():
    # 初始化out.txt文件
    with open("out.txt", "w", encoding="utf-8") as f:
        f.write("YOLOv8 目标检测结果记录\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
    
    parser = argparse.ArgumentParser(description='YOLOv8 目标检测程序')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径 (.pt)')
    parser.add_argument('--source', type=str, help='输入源: 图片/视频路径或 "camera"')
    parser.add_argument('--conf', type=float, default=0.30, help='置信度阈值 (默认: 0.30)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
        
    # 创建检测器
    detector = YOLODetector(args.model)
    
    # 根据输入源执行不同的检测任务
    if args.source:
        if args.source.lower() == 'camera':
            # 摄像头检测
            detector.detect_camera(conf_threshold=args.conf)
        elif os.path.isfile(args.source):
            # 文件检测
            if args.source.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 图片检测
                detector.detect_image(args.source, conf_threshold=args.conf)
            elif args.source.lower().endswith(('.mp4', '.avi', '.mov')):
                # 视频检测
                detector.detect_video(args.source, conf_threshold=args.conf)
            else:
                print("不支持的文件格式，请使用图片(.png, .jpg, .jpeg)或视频(.mp4, .avi, .mov)文件")
        else:
            print("输入源不存在，请检查路径是否正确")
    else:
        print("请指定输入源 (--source) 或使用 --help 查看帮助")

if __name__ == "__main__":
    main()