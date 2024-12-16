# Source-Free Domain-Invariant Performance Prediction

Welcome to the repository for the **ECCV 2024** paper.

📄 **[Read the Paper](https://arxiv.org/pdf/2408.02209)**  
🖼️ **[View the Poster](logs/eccv2024_poster.jpg)**  

This work focuses on predicting model performance on a target domain **without access to the source data**.

---

## 📋 Method Overview  

The method predicts classifier accuracy on the target domain through the following steps:  
1. **Calculate GMM Statistics**:  
   Compute the mean and covariance of the logits for each class based on the target predictions.  
2. **Calibration**:  
   Replace Softmax with a GMM-based generative model to calibrate the final probabilities of the test sample.
3. **Assess the Correctness of the Prediction**:  
   Calculate 2 losses: CELoss to one-hot encoding for the most probable class, and CELoss to a uniform distribution (random prediction).
   Backpropogate through only the last layer (aka classifier) and compare gradient norms to determine correctness of prediction:  
   - If the gradient norm to the predicted class <= gradient norm to a random class → **Correct**.  
   - Otherwise → **Incorrect**.  

---


## 🔧 Installation

Before running the code, install the required dependencies using:  
```bash
pip install -r requirements.txt
```

## 🚀 Reproducing Results  



### Step 1: Download Pretrained Model  
- Download the MNIST-trained model [lenet_mnist.pt](https://drive.google.com/file/d/1MN60HAsEjw4EU_NWhDi1x_nJxOVRS4lo/) and place it in the `logs/` folder.  

### Step 2: Run the Experiment  
Run the following command to reproduce results on the **SVHN** and **USPS** datasets:  
``` 
python main.py
```

This code will:
- Automatically download the SVHN and USPS datasets and save them in the `logs/` folder.
- Store the features of the target datasets in the `logs/` folder.
- Predict the model’s performance on each target dataset.

### Additional Experiments
Explore the `all_experiments/` folder for other experiments and baseline comparisons.

## 📖 Citation
If you use this code, please cite our paper:

```
@inproceedings{khramtsova2025sourcefreepp,
title={Source-Free Domain-Invariant	Performance Prediction},
author={Khramtsova, Ekaterina and Baktashmotlagh, Mahsa  and Zuccon, Guido and Wang, Xi and Salzmann, Mathieu},
editor={Leonardis, Ale{\v{s}} and Ricci, Elisa and Roth, Stefan and Russakovsky, Olga and Sattler, Torsten and Varol, G{\"u}l},
booktitle={Proceedings of The 18th European Conference on Computer Vision (ECCV)},
year={2024},
publisher={Springer Nature Switzerland},
pages={99--116},
isbn={978-3-031-72989-8}
}
```
