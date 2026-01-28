# โครงสร้างProject

my-crypto-project/
│
├── .gitignore                 # Prevents large/private files from being uploaded
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── data/                      # Dataset (not tracked by Git)
│   ├── raw/                   # Original Kaggle data
│   └── processed/             # Cleaned data from P1
│
├── notebooks/                 # Jupyter notebooks
│   ├── 1.0-data-cleaning.ipynb
│   ├── 1.1-graph-features.ipynb
│   ├── 2.0-model-training.ipynb
│   └── 2.1-model-evaluation.ipynb
│
├── src/                       # Reusable Python modules
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

    suphakit
