Vision Transformer Comparison for Malware Image Classification
Proyek ini mengimplementasikan dan membandingkan tiga arsitektur Vision Transformer untuk klasifikasi 25 jenis malware family berdasarkan visualisasi biner mereka.
Model yang Dibandingkan:
| Model | Input Size | Parameters | Kecepatan Inference |
|-------|-----------|------------|---------------------|
| **ViT Base** | 224×224 | ~86M | ~15-20ms/image |
| **DeiT Small** | 224×224 | ~22M | ~10-15ms/image |
| **SwinV2 Base** | 256×256 | ~88M | ~20-25ms/image |
 Cara Menjalankan di Kaggle

**Step 1: Akses Dataset**

1. **Buka link dataset**: https://www.kaggle.com/datasets/ikrambenabd/malimg-original

2. **Add dataset ke notebook Anda:**
   - Klik tombol **"New Notebook"** di bagian kanan atas halaman dataset
   - Atau klik **"+ Add Data"** jika sudah memiliki notebook
---

### **Step 2: Buat Notebook Baru**

1. **Pilih opsi "Code"** di halaman dataset
2. **Klik "Create New Notebook"**
3. Notebook baru akan terbuka dengan dataset sudah ter-attach
---
### **Step 3: Aktifkan GPU T4x2**

1. **Klik tombol "Settings"**  di panel kanan notebook

2. **Di bagian "Accelerator"**:
   - Pilih **"GPU T4 x2"** dari dropdown
   - Jika tidak tersedia, pilih **"GPU T4"** saja (single GPU)
   

3. **Klik "Save"** untuk apply perubahan

4. **Verifikasi GPU aktif:**
```python
   # Jalankan cell ini untuk cek GPU
   import torch
   print(f"CUDA Available: {torch.cuda.is_available()}")
   print(f"GPU Count: {torch.cuda.device_count()}")
   if torch.cuda.is_available():
       print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```
   
   **Output yang diharapkan:**
```
   CUDA Available: True
   GPU Count: 2
   GPU Name: Tesla T4
```

---

### **Step 4: Install Dependencies**

**Buat cell baru** dan jalankan:
```python
# Install timm library (untuk pretrained models)
!pip install timm --quiet

# Verify installation
import timm
print(f"✓ timm version: {timm.__version__}")
```

**Output:**
```
✓ timm version: 0.9.12
```

---

### **Step 5: Verify Dataset Path**

Dataset otomatis ter-mount di `/kaggle/input/`. **Verify path dengan cara:**
```python
import os

# Path dataset di Kaggle
DATA_DIR = "/kaggle/input/malimg-original/malimg_paper_dataset_imgs"

# Check apakah path exists
if os.path.exists(DATA_DIR):
    print(f"✓ Dataset found at: {DATA_DIR}")
    
    # List folders (should show 25 malware families)
    folders = sorted(os.listdir(DATA_DIR))
    print(f"✓ Number of malware families: {len(folders)}")
    print(f"✓ First 5 families: {folders[:5]}")
else:
    print(f"✗ Dataset NOT found at: {DATA_DIR}")
    print("Available paths:")
    print(os.listdir("/kaggle/input/"))
```

**Output yang diharapkan:**
```
✓ Dataset found at: /kaggle/input/malimg-original/malimg_paper_dataset_imgs
✓ Number of malware families: 25
✓ First 5 families: ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J']
```

---

### **Step 6: Copy Source Code**
Copy semua kodingan yang sudah tertera pada kaggle.

**Step 1: Prerequisites**

Pastikan sudah terinstall:
- Python 3.8 - 3.10
- CUDA Toolkit 11.8 atau 12.1 (untuk GPU)
- Git
- Visual Studio Code

---

### **Step 2: Download Repository**

#### **Opsi A: Clone via Git**
```bash
# Buka terminal/command prompt
cd Desktop  # atau folder lain sesuai keinginan

# Clone repository
git clone https://github.com/username/malware-vit-comparison.git

# Masuk ke folder
cd malware-vit-comparison
```

#### **Opsi B: Download ZIP**

1. Buka repository GitHub
2. Klik tombol **"Code"** → **"Download ZIP"**
3. Extract ZIP file ke folder yang diinginkan
4. Buka folder di terminal

---

### **Step 3: Download Dataset**

#### **Cara 1: Download Manual dari Kaggle**

1. **Buka link dataset**: https://www.kaggle.com/datasets/ikrambenabd/malimg-original

2. **Klik tombol "Download"**  di kanan atas

3. **Extract file `archive.zip`:**
```bash
   # Windows (PowerShell)
   Expand-Archive -Path "C:\Users\YourName\Downloads\archive.zip" -DestinationPath ".\data"
   
   # Linux/Mac
   unzip ~/Downloads/archive.zip -d ./data
```

4. **Verify struktur folder:**
```
   malware-vit-comparison/
   ├── data/
   │   └── malimg_paper_dataset_imgs/
   │       ├── Adialer.C/
   │       ├── Agent.FYI/
   │       └── ... (25 folders total)
   ├── vit_model.py
   ├── deit_model.py
   └── swinv2_model.py
```

#### **Cara 2: Download via Kaggle API**
```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials (butuh API token dari https://www.kaggle.com/settings)
# Download kaggle.json dan letakkan di:
# Windows: C:\Users\\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d ikrambenabd/malimg-original

# Extract
unzip malimg-original.zip -d ./data
```

---

### **Step 4: Setup Virtual Environment**

#### **Opsi A: Menggunakan Conda (Recommended)**
```bash
# Create environment
conda create -n malware-vit python=3.10 -y

# Activate environment
conda activate malware-vit

# Install PyTorch dengan CUDA (pilih sesuai GPU Anda)
# Untuk CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Untuk CUDA 12.1:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Untuk CPU only (tanpa GPU):
conda install pytorch torchvision cpuonly -c pytorch -y
```

#### **Opsi B: Menggunakan venv (Python Built-in)**
```bash
# Create environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install PyTorch (sesuaikan dengan sistem Anda)
# Cek command di: https://pytorch.org/get-started/locally/

# Contoh untuk CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### **Step 5: Install Dependencies**
```bash
# Install dari requirements.txt
pip install -r requirements.txt

# Atau install manual
pip install timm pandas matplotlib seaborn scikit-learn
```

**Verify installation:**
```bash
python check_installation.py
```

**Output yang diharapkan:**
```
✓ torch installed
✓ torchvision installed
✓ timm installed
✓ numpy installed
✓ pandas installed
✓ matplotlib installed
✓ seaborn installed
✓ sklearn installed
✓ PIL installed

✓ All packages installed successfully!

PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
```

---

### **Step 6: Buka Project di VS Code**
### **Step 7: Update Path Dataset**
**PENTING:** Anda harus mengubah `DATA_DIR` di setiap script untuk local path.

#### **File: `vit_model.py`**

Cari baris ini (sekitar baris 20):
```python
#  KAGGLE PATH (JANGAN DIGUNAKAN DI LOCAL)
DATA_DIR = "/kaggle/input/malimg-original/malimg_paper_dataset_imgs"
```

**Ganti dengan:**
```python
#  LOCAL PATH (RELATIVE)
DATA_DIR = "./data/malimg_paper_dataset_imgs"

# Atau gunakan ABSOLUTE PATH:
# Windows:
# DATA_DIR = "C:/Users/YourName/Desktop/malware-vit-comparison/data/malimg_paper_dataset_imgs"
# Linux/Mac:
# DATA_DIR = "/home/username/Desktop/malware-vit-comparison/data/malimg_paper_dataset_imgs"
```

#### **File: `deit_model.py`**

Cari baris ini (sekitar baris 18):
```python
#  KAGGLE PATH
DATA_DIR = "/kaggle/input/malimg-original/malimg_paper_dataset_imgs"
OUTPUT_DIR = "/kaggle/working/"
```

**Ganti dengan:**
```python
#  LOCAL PATH
DATA_DIR = "./data/malimg_paper_dataset_imgs"
OUTPUT_DIR = "./results/deit/"

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

#### **File: `swinv2_model.py`**

Cari baris ini (sekitar baris 18):
```python
#  KAGGLE PATH
DATA_DIR = "/kaggle/input/malimg-original/malimg_paper_dataset_imgs"
OUTPUT_DIR = "/kaggle/working/"
```

**Ganti dengan:**
```python
#  LOCAL PATH
DATA_DIR = "./data/malimg_paper_dataset_imgs"
OUTPUT_DIR = "./results/swinv2/"

# Buat folder output
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

### **Step 8: Verify Dataset Path**

**Buat file test**: `test_path.py`
```python
import os

# Test path
DATA_DIR = "./data/malimg_paper_dataset_imgs"

print(f"Testing path: {DATA_DIR}")
print(f"Path exists: {os.path.exists(DATA_DIR)}")

if os.path.exists(DATA_DIR):
    folders = sorted(os.listdir(DATA_DIR))
    print(f"Number of folders: {len(folders)}")
    print(f"First 5 folders: {folders[:5]}")
    
    # Count total images
    total_images = 0
    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            num_images = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
            print(f"  {folder}: {num_images} images")
            total_images += num_images
    
    print(f"\n✓ Total images: {total_images}")
else:
    print("✗ Path does NOT exist!")
    print("\nCurrent directory:", os.getcwd())
    print("Contents:", os.listdir('.'))
```

**Run test:**
```bash
python test_path.py
```

**Output yang diharapkan:**
```
Testing path: ./data/malimg_paper_dataset_imgs
Path exists: True
Number of folders: 25
First 5 folders: ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J']
  Adialer.C: 122 images
  Agent.FYI: 116 images
  ...
  
✓ Total images: 9339
```

---
### **Step 9: Run Training di VS Code**

#### **Opsi A: Via Terminal**
```bash
# Activate environment dulu
conda activate malware-vit

# Run ViT
python vit_model.py

# Run DeiT (setelah ViT selesai)
python deit_model.py

# Run SwinV2 (setelah DeiT selesai)
python swinv2_model.py
```

#### **Opsi B: Via VS Code Run Button**

**Buka file** (e.g., `vit_model.py`)
---
### **Step 10: Acces Result**
Hasil training akan tersimpan di folder `results/`:
```
malware-vit-comparison/
├── results/
│   ├── deit/
│   │   ├── deit_malimg_best.pth
│   │   ├── DeiT_confusion_matrix.png
│   │   └── ...
│   └── swinv2/
│       ├── swinv2_malimg_256.pth
│       ├── SwinV2_256_confusion_matrix.png
│       └── ...
└── vit_malimg_optimized.pth  # ViT results di root folder
```

**Open results via VS Code:**

1. **Klik icon "Explorer"**  di sidebar kiri
2. **Navigate ke folder `results/`**
3. **Double-click file `.png`** untuk lihat visualisasi
4. **Right-click `.csv`** → **"Open Preview"** untuk lihat metrics


