# โครงสร้างProject

my-crypto-project/
│
├── .gitignore               <-- สำคัญมาก! บอก Git ว่าห้ามอัปโหลดไฟล์ไหน (เช่น ไฟล์ข้อมูลขนาดใหญ่)
├── README.md                <-- หน้าปกโปรเจกต์ (ชื่อกลุ่ม, วิธีรัน, อธิบายโปรเจกต์)
├── requirements.txt         <-- รายชื่อ Library ที่ต้องใช้ (pandas, streamlit, networkx)
│
├── data/                    <-- เก็บข้อมูล (แต่ห้ามอัปขึ้น Git ให้อ่านจากเครื่องใครเครื่องมัน)
│   ├── raw/                 <-- ข้อมูลดิบที่โหลดจาก Kaggle (ห้ามแก้ไข)
│   └── processed/           <-- ข้อมูลที่ P1 Clean แล้ว (พร้อมให้ P2 เอาไปเทรน)
│
├── notebooks/               <-- พื้นที่ทดลอง (Sandboxes) ของ P1 และ P2
│   ├── 1.0-data-cleaning.ipynb    <-- ของ P1
│   ├── 1.1-graph-features.ipynb   <-- ของ P1
│   ├── 2.0-model-training.ipynb   <-- ของ P2
│   └── 2.1-model-evaluation.ipynb <-- ของ P2
│
├── src/                     <-- เก็บ Code ภาษา Python ที่เขียนเป็นฟังก์ชัน (Re-usable code)
│   ├── __init__.py
│   ├── features.py          <-- ฟังก์ชันสร้าง Graph Feature (ดึงมาจาก Notebook P1)
│   └── models.py            <-- ฟังก์ชันเทรนโมเดล (ดึงมาจาก Notebook P2)
│
├── models/                  <-- เก็บไฟล์โมเดลที่เทรนเสร็จแล้ว
│   └── xgboost_v1.pkl       <-- ไฟล์ที่ P2 ส่งให้ P3 ใช้
│
├── app/                     <-- พื้นที่ทำงานหลักของ P3 (Web App)
│   ├── main.py              <-- ไฟล์หลักสำหรับรัน Streamlit
│   ├── utils.py             <-- ฟังก์ชันช่วยวาดกราฟบนเว็บ
│   └── assets/              <-- เก็บรูปภาพ Logo, CSS
│
└── reports/                 <-- เก็บเล่มรายงานและสไลด์
    └── final_report.pdf