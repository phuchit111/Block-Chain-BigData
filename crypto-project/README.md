# Crypto Transaction Analysis Project

โปรเจคนี้ใช้วิเคราะห์ธุรกรรม Bitcoin จาก Elliptic Dataset เพื่อตรวจจับธุรกรรมที่ผิดกฎหมาย

## 📋 ข้อกำหนดเบื้องต้น

- Python 3.8 ขึ้นไป
- pip (Python package manager)

## 🚀 วิธีการรันโปรเจค

### 1. ติดตั้ง Dependencies

```bash
cd /Users/annopsangsila/Desktop/BigData/crypto-project
pip install -r requirements.txt
```

หรือถ้าใช้ virtual environment (แนะนำ):

```bash
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน virtual environment
source venv/bin/activate  # สำหรับ Mac/Linux

# ติดตั้ง dependencies
pip install -r requirements.txt
```

### 2. เตรียมข้อมูล

ข้อมูลอยู่ใน folder `data/`:
- `data/raw/` - ข้อมูลต้นฉบับจาก Kaggle
- `data/processed/` - ข้อมูลที่ผ่านการประมวลผลแล้ว

### 3. รัน Jupyter Notebooks (ถ้ามี)

```bash
# ติดตั้ง Jupyter
pip install jupyter

# เปิด Jupyter Notebook
jupyter notebook
```

จากนั้นเปิดไฟล์ในโฟลเดอร์ `notebooks/`:
- `1.0-data-cleaning.ipynb` - ทำความสะอาดข้อมูล
- `1.1-graph-features.ipynb` - สร้าง graph features
- `2.0-model-training.ipynb` - ฝึก model
- `2.1-model-evaluation.ipynb` - ประเมินผล model

### 4. รัน Streamlit Web Application

```bash
# ติดตั้ง Streamlit
pip install streamlit

# รัน web app
streamlit run app/main.py
```

เว็บแอปจะเปิดที่ `http://localhost:8501`

## 📁 โครงสร้างโปรเจค

```
crypto-project/
│
├── .gitignore                 # ป้องกันไฟล์ขนาดใหญ่/ข้อมูลส่วนตัว
├── README.md                  # เอกสารโปรเจค
├── requirements.txt           # Python dependencies
│
├── data/                      # Dataset
│   ├── raw/                   # ข้อมูลต้นฉบับจาก Kaggle
│   └── processed/             # ข้อมูลที่ทำความสะอาดแล้ว
│
├── notebooks/                 # Jupyter notebooks
│   ├── 1.0-data-cleaning.ipynb
│   ├── 1.1-graph-features.ipynb
│   ├── 2.0-model-training.ipynb
│   └── 2.1-model-evaluation.ipynb
│
├── src/                       # Python modules
│   ├── __init__.py
│   ├── features.py            # Graph feature engineering
│   └── models.py              # Model training and prediction
│
├── models/                    # Trained models
│   └── xgboost_v1.pkl
│
├── app/                       # Streamlit Web Application
│   ├── main.py
│   ├── utils.py
│   └── assets/
│
└── reports/
    └── final_report.pdf
```

## 📊 ข้อมูลที่ใช้

- **elliptic_txs_features.csv** - Features ของธุรกรรม Bitcoin (689 MB)
- **elliptic_bitcoin_dataset_v0.csv** - Dataset เวอร์ชัน 0 (150 MB)
- **elliptic_bitcoin_dataset_v1_graph.csv** - Dataset เวอร์ชัน 1 พร้อม graph structure (150 MB)

## 🛠️ การพัฒนา

### ใช้งาน Python modules

```python
from src.features import *
from src.models import *
```

### ฝึก Model ใหม่

ดูตัวอย่างใน `notebooks/2.0-model-training.ipynb`

## 📝 หมายเหตุ

- ตรวจสอบว่าไฟล์ `requirements.txt` มี dependencies ที่จำเป็นครบถ้วน
- ถ้ายังไม่มี notebooks ให้สร้างขึ้นมาใหม่ตามโครงสร้างที่กำหนด
- Model ที่ฝึกแล้วจะถูกบันทึกใน `models/`

## 👥 ทีมพัฒนา

- Suphakit
