# 🔫 Custom Guns Object Detection

## 📘 Overview
This project focuses on building a **Custom Object Detection System** for identifying and classifying various gun types using **PyTorch**. The project integrates **TensorBoard** for experiment tracking and **DVC (Data Version Control)** for pipeline management, ensuring reproducibility and scalability. The deployment-ready API is built using **FastAPI** and documented using **Swagger UI** and **Postman**.

---

## 🧱 Project Structure
```
Custom_Guns_Object_Detection/
│
├── .dvc/                     # DVC metadata and tracking files
├── artifacts/                # Stores trained models, metrics, and results
├── config/                   # YAML/JSON configuration files for training & data
├── logs/                     # Logging directory for monitoring runs
├── notebooks/                # Jupyter notebooks for experimentation
├── src/                      # Core source code for model, data, and utils
├── tensorboard_logs/         # TensorBoard logs for visualization
├── utils/                    # Utility functions for data and model ops
│
├── dvc.yaml                  # DVC pipeline configuration
├── dvc.lock                  # DVC version lock file
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation script
├── main.py                   # Main execution script
└── README.md                 # Project documentation
```

---

## ⚙️ Features
- **Custom Object Detection** using PyTorch.
- **Modular ML pipeline** for data ingestion, processing, and training.
- **Experiment tracking** with TensorBoard.
- **Data and code versioning** using DVC & Git.
- **FastAPI-based REST API** for model inference.
- **Swagger UI** & **Postman** testing support.

---

## 🧪 Workflow
1. **Project Setup** – Environment, dependencies, and folder structure.  
2. **Data Ingestion (Kaggle)** – Fetch and prepare datasets.  
3. **Data Processing** – Apply augmentations, normalization, and label encoding.  
4. **Model Architecture** – Define and build a custom CNN using PyTorch.  
5. **Model Training** – Train and validate with TensorBoard metrics.  
6. **Experiment Tracking** – Log metrics, loss curves, and results via TensorBoard.  
7. **Pipeline Management (DVC)** – Automate stages for ingestion, processing, and training.  
8. **API Deployment** – Serve model predictions with FastAPI.

---

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/Sumit-Prasad01/Custom-Guns-Object-Detection.git
cd Custom-Guns-Object-Detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Usage
### Run the pipeline with DVC
```bash
dvc repro
```

### Launch training manually
```bash
python main.py
```

### View TensorBoard logs
```bash
tensorboard --logdir=tensorboard_logs
```

### Run FastAPI app
```bash
uvicorn main:app --reload
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI.

---

## 📊 Experiment Tracking (TensorBoard)
All training runs are logged in `tensorboard_logs/`. You can visualize:  
- Training & validation loss curves  
- Model accuracy trends  
- Learning rate & optimization metrics  

---

## 🧰 Technologies Used
- **PyTorch** – Deep Learning Framework  
- **TensorBoard** – Visualization and Experiment Tracking  
- **DVC** – Pipeline and Data Versioning  
- **FastAPI** – API Deployment  
- **Kaggle API** – Dataset Integration  
- **Docker** *(optional)* – Containerized environment  

---

## 📈 Future Enhancements
- Integrate **MLflow** for advanced experiment management  
- Deploy model on **Google Cloud Run** or **AWS Lambda**  
- Add **real-time inference** via webcam stream  
- Expand dataset for multi-class object detection  

---
