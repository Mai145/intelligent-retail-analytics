Intelligent Retail Analytics System 🛒📊
Bu proje, perakende mağazaları için geliştirilmiş, YOLOv8 ve DeepFace tabanlı bir müşteri davranış analiz sistemidir. RTX 4060 GPU gücünü kullanarak çoklu müşteri takibi ve duygu analizi gerçekleştirir.

✨ Öne Çıkan Özellikler
Multi-Person Tracking: Her müşteriye benzersiz bir ID atayarak hareket takibi yapar.

GPU Hızlandırma: NVIDIA CUDA çekirdekleri ile gerçek zamanlı (FPS) analiz sağlar.

Duygu Filtreleme: Yanlış pozitifleri (masa, sandalye vb.) eleyen ve "Sad/Fear" hatalarını minimize eden özel güven eşiği (Confidence Threshold) mekanizması.

Otomatik Raporlama: Analiz sonuçlarını data/reports/ klasörüne zaman damgalı .csv olarak kaydeder.

🛠️ Teknolojiler
Python 3.12+

YOLOv8: Nesne tespiti ve tracking.

DeepFace: Duygu tanıma (Emotion Recognition).

TensorFlow & PyTorch: Derin öğrenme motorları.

OpenCV: Görüntü işleme ve görselleştirme.

🚀 Sıfırdan Kurulum Rehberi
Projeyi başka bir bilgisayarda çalıştırmak için şu adımları izleyin:

1. Depoyu Klonlayın ve Ortamı Hazırlayın
Bash
git clone https://github.com/KULLANICI_ADIN/intelligent-retail-analytics.git
cd intelligent-retail-analytics
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash için
2. Gerekli Kütüphaneleri Yükleyin
Bash
# Temel kütüphaneler
pip install ultralytics deepface opencv-python tensorflow==2.18.1 lapx filterpy tf-keras

# GPU (RTX Serisi) Desteği İçin:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. Donanım Gereksinimi
GPU: NVIDIA RTX 30/40 Serisi (Önerilen).

Sürücü: Güncel NVIDIA Game Ready/Studio Driver.

CUDA: Toolkit 11.8.

4. Çalıştırma
Analiz edilecek videoyu data/ klasörüne ekledikten sonra:

Bash
python src/main.py
📂 Dosya Yapısı
Plaintext
intelligent-retail-analytics/
├── src/                # Kaynak kodlar (main.py)
├── data/               # Videolar (.mp4)
│   └── reports/        # Otomatik oluşturulan CSV raporları
├── venv/               # Sanal ortam dosyaları
├── yolov8n.pt          # YOLO model ağırlıkları
└── README.md           # Proje dökümantasyonu