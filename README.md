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

1. **Klik tombol "Settings"** (⚙️) di panel kanan notebook

2. **Di bagian "Accelerator"**:
   - Pilih **"GPU T4 x2"** dari dropdown
   - Jika tidak tersedia, pilih **"GPU T4"** saja (single GPU)
   
   ![GPU Settings](https://i.imgur.com/example3.png)

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
