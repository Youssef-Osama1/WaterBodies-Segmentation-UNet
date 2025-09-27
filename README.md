# 🌊 Water Body Segmentation from Satellite Images using Multispectral & Optical Data

## 📌 Overview  
This project focuses on **semantic segmentation of water bodies** using multispectral satellite imagery.  
Accurate water segmentation is crucial for:  
- 🌍 Water resources monitoring  
- 🌊 Flood management  
- 🌱 Environmental conservation

We experiment with two approaches:  
1. **Custom U-Net (from scratch)**  
2. **U-Net with Pretrained Encoder (ResNet34)**  
   - **Model A:** Trained on **all 12 bands**  
   - **Model B:** Trained on **3 bands (NIR, SWIR1, SWIR2)** with `encoder_weights="imagenet"`

---

## 📂 Project Structure
- **notebooks/** → contains two Jupyter notebooks:  
  - `Water_Segmentation_From_Scratch.ipynb`: implements a U-Net model trained from scratch using all 12 spectral bands.  
  - `Water_Segmentation_With_Pretrained.ipynb`: experiments with pretrained encoders:  
    - **Model A:** U-Net trained on all 12 bands with `encoder_weights=None` (encoder initialized from scratch).  
    - **Model B:** U-Net trained on 3 bands with `encoder_weights="imagenet"`.  
- **task.pdf** → Task description and project requirements.  
- **README.md** → Project documentation.  

---

## ⚙️ Dataset & Preprocessing  

We use the **Harmonized Sentinel-2 / Landsat dataset**, which provides 12-band multispectral patches along with binary water masks.  

- **Input:** 128×128×12 patches  
- **Output:** 128×128×1 binary masks (water vs non-water)  

![Spectral Bands](assets/Satellite%20Bands.jpeg)  

### 🔄 Preprocessing Steps  
- Per-channel standardization  
- Patch extraction (128×128)  
- Train/Validation/Test split  

---

## 🧠 Models  

### 🔹 1. Custom U-Net  
- Implemented from scratch with **encoder-decoder + skip connections**.  
- Trained for **25 epochs**.  
- Achieved:  
  - **Test IoU ≈ 0.74**  
  - **Test F1-Score ≈ 0.85**  

### 🔹 2. U-Net + ResNet34 (Pretrained)  

- **Model A (All 12 Bands)**  
  - Encoder initialized with **random weights** (since ImageNet pretraining is RGB).  
  - Achieved:  
    - **Test IoU ≈ 0.77**  
    - **Test F1-Score ≈ 0.86**  

- **Model B (3 Bands: NIR, SWIR1, SWIR2)**  
  - Encoder initialized with **ImageNet weights** (`encoder_weights="imagenet"`).  
  - Achieved:  
    - **Test IoU ≈ 0.84**  
    - **Test F1-Score ≈ 0.91**  

---

## 📊 Results  

| Model                  | Bands Used   | IoU   | F1-Score | Precision | Recall |
|-------------------------|-------------|-------|----------|-----------|--------|
| Custom U-Net (scratch) | 12 bands    | 0.74  | 0.85     | 0.92      | 0.79   |
| U-Net + ResNet34 (A)   | 12 bands    | 0.77  | 0.86     | 0.94      | 0.80   |
| U-Net + ResNet34 (B)   | 3 bands     | 0.84  | 0.91     | 0.91      | 0.90   |

---

## 🖼️ Visualization  
![Prediction Example](assets/Prediction%20Example1.png)
![Prediction Example](assets/Prediction%20Example2.png)
![Prediction Example](assets/Prediction%20Example3.png)
![Prediction Example](assets/Prediction%20Example4.png)
