# Water Body Segmentation from Satellite Images

This project applies **deep learning** techniques to segment water bodies from **multispectral satellite imagery**.  
We use a **U-Net architecture built from scratch** to train and evaluate the model on 12-band satellite images.

---

## 📂 Project Structure
- **notebooks/** → Jupyter notebooks containing the code for data preprocessing, model training, and evaluation.  
- **task.pdf** → Task description and project requirements.  
- **README.md** → Project documentation.  

---

## ⚙️ Methodology
1. **Data Preparation**  
   - Extracted multispectral TIFF images and corresponding binary masks.  

2. **Data Normalization (Per-Channel Normalization)**  
   - Standardized each spectral band separately by subtracting its mean and dividing by its standard deviation.  

3. **Data Visualization**  
   - Visualized the 12 spectral bands individually.  
   - Displayed example binary masks corresponding to the images.  

   ![Satellite Bands](assets/Satellite%20Bands.jpeg)

4. **Model Architecture and Training**  
   - Implemented a **U-Net** with encoder–decoder structure.  
   - Trained the model from scratch on 128×128 patches with 12 channels.  

5. **Evaluation (IoU, Precision, Recall, and F1-score)**  
   - Evaluated model predictions against ground truth masks.  

---

## 📊 Results
Final model performance on the test set:  
- **IoU:** 0.7452  
- **Precision:** 0.9268  
- **Recall:** 0.7918  
- **F1-score:** 0.8540  

Example of prediction vs ground truth:  

![Prediction Example](assets/Prediction%20Example.png)

---

## 🚀 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, OpenCV, scikit-learn  
- Matplotlib
