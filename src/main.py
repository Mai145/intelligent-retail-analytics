import os
import cv2
import csv
import numpy as np
from datetime import datetime
from collections import Counter, deque

# 1. GPU VE LOG AYARLARI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# GPU bellek yönetimi
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(">>> RTX 4060 Hazir ve Nazir.")
    except: pass

from ultralytics import YOLO
from deepface import DeepFace

class RetailAnalyticsSystem:
    def __init__(self, video_filename):
        # Klasör Yolları (Sabitlendi)
        self.base_dir = r"C:\intelligent-retail-analytics"
        self.video_path = os.path.join(self.base_dir, "data", video_filename)
        
        report_dir = os.path.join(self.base_dir, "data", "reports")
        if not os.path.exists(report_dir): os.makedirs(report_dir)
        self.report_path = os.path.join(report_dir, f"rapor_{datetime.now().strftime('%H%M%S')}.csv")

        # Modeller (YOLOv8 Nano)
        self.model = YOLO("yolov8n.pt") 
        self.cap = cv2.VideoCapture(self.video_path)
        
        self.track_history = {}
        self.frame_skip = 2 # Performansli analiz
        self.frame_count = 0

        # CSV Dosyasini olustur
        with open(self.report_path, mode='w', newline='') as f:
            csv.writer(f).writerow(['Zaman', 'Musteri_ID', 'Duygu'])

    def run(self):
        print(f">>> Analiz Basliyor. Cikis icin 'q'ya basin.")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            self.frame_count += 1
            # Görüntü Boyutu
            frame = cv2.resize(frame, (1080, 720))

            # --- YOLO TAKİP (Sadece Person Sınıfı) ---
            results = self.model.track(frame, persist=True, classes=[0], device=0, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    
                    # Yüz Bölgesi (Üst %35)
                    face_roi = frame[max(0, y1):y1+int((y2-y1)*0.35), max(0, x1):x2]

                    if self.frame_count % self.frame_skip == 0 and face_roi.size > 0:
                        try:
                            # --- DEEPFACE ANALİZ (MediaPipe Bagimliligi Yok!) ---
                            analysis = DeepFace.analyze(face_roi, 
                                                        actions=['emotion'], 
                                                        enforce_detection=True, 
                                                        detector_backend='opencv', # Kritik: OpenCV kullanir
                                                        silent=True)
                            
                            emotion = analysis[0]['dominant_emotion'].upper()
                            
                            if track_id not in self.track_history:
                                self.track_history[track_id] = deque(maxlen=10)
                            self.track_history[track_id].append(emotion)
                            
                            # CSV Loglama
                            with open(self.report_path, mode='a', newline='') as f:
                                csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), track_id, emotion])
                        except:
                            # Yüz bulunamazsa (veya masa/sandalye ise) hata vermeden gecer
                            pass

                    # Görselleştirme
                    label = "Scanning..."
                    if track_id in self.track_history and self.track_history[track_id]:
                        label = Counter(self.track_history[track_id]).most_common(1)[0][0]
                    
                    # Kutucuk ve Yazı
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                    cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            
            cv2.imshow('Kadir Has MIS - Retail Analytics (GPU)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Bitti. Raporun burada: {self.report_path}")

if __name__ == "__main__":
    app = RetailAnalyticsSystem('classroom.mp4')
    app.run()