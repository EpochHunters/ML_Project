# EEG Classification and Image Visualization Project

## Table of Contents  
- [Project Overview](#project-overview)  
- [Outcomes](#outcomes)  
- [Approaches for Image Generation](#approaches-for-image-generation)  
- [Advancements in BCI Technology](#advancements-in-bci-technology)  
- [Contributions](#contributions)  

---

## Project Overview  
This project focuses on classifying EEG signals and reconstructing visual stimuli from them using deep learning models. By exploring the temporal and spatial patterns of brain activity, we developed multiple architectures, including CNN-LSTM hybrids and encoder-based models, to improve classification accuracy. Additionally, we implemented spectrally normalized GANs (SNGAN) and conditional GANs (CGAN) to generate images based on EEG embeddings.

---

## Outcomes  

### 1. **Improved Classification Accuracy**  
We achieved a classification accuracy of **93%** for EEG signals across 40 classes by redefining and refining the encoder architecture. The encoder-based model emerged as the most effective, particularly for recognizing spatial and temporal patterns in EEG data.  

### 2. **Feature Representation**  
By leveraging temporal, spatial, and residual blocks in the encoder, we extracted refined features from EEG signals, resulting in a **128-dimensional latent feature vector**. This vector was then used as input for the GAN models.  

### 3. **Models Used**  
Several architectures were implemented and compared, including:  
- **LSTM-CNN**  
- **1D-CNN-LSTM**  
- **2D-CNN-LSTM**  
- **Deep CNN**  
- **Encoder-based architectures**  

The encoder-based model demonstrated superior performance for EEG signal classification, while the **CNN architecture** showed promising results for image classification tasks.  

### 4. **SNGAN and CGAN for Image Generation**  
- We used the latent feature vector (augmented with noise) as input to the **SNGAN** generator to reconstruct images with a resolution of **128 × 128 pixels**, representing the visual stimuli presented to subjects during EEG data collection.  
- Although these approaches were promising, neither SNGAN nor CGAN produced conclusive results, highlighting challenges in accurately reconstructing visual stimuli from EEG data.  

---

## Approaches for Image Generation  

1. **Standard Training Across All Subjects**:  
   - EEG embeddings from all six subjects were used to train the GAN architecture, ensuring sufficient variance and an unbiased data spread.  
   - However, the lack of personalization led to challenges in capturing specific EEG patterns associated with individual subjects.  

2. **Subject-Specific Training**:  
   - GANs were trained on EEG embeddings from **a single subject**, accounting for unique personal experiences and visual stimuli.  
   - This approach improved the likelihood of capturing consistent patterns in EEG data for each class of images.  

3. **Feature Vector-Based Training**:  
   - A ResNet50 model was trained to extract feature vectors from visual stimuli, which were then used as input for the GAN.  
   - This approach aimed to generate images more closely aligned with the original class of visual stimuli provided to subjects.  

---

## Advancements in BCI Technology  
This project contributes to advancements in **brain-computer interface (BCI)** technologies by:  
1. Enhancing EEG signal decoding for improved classification and understanding of brain activity.  
2. Exploring the reconstruction of visual stimuli from EEG embeddings, opening avenues for applications in neuroscience, cognitive research, and assistive technologies.  

---

## Contributions  

### **Team Members and Responsibilities**  
- **Maneesh Pulidindi 12241380**:  
  - Worked on preprocessing EEG signals.  
  - Implemented state diffusion models, fine-tuned to use EEG embeddings instead of text.  

- **Anshal Khatri M24DS001**:  
  - Developed and fine-tuned the Bi-LSTM-CNN-GAN model.  
  - Created the project's documentation.  

- **Byomakesh Panda M24DS004**:  
  - Implemented a two-headed model for simultaneous classification and image generation using LSTM and GANs.  
  - Extracted image feature vectors via **transfer learning** with ResNet18 and used these vectors as latent input for the GAN.  

- **Chinmay Bakhale M24DS005**:  
  - Developed an innovative approach using images to generate latent vectors for training GANs.  

- **Raghav Borikar M24DS010**:  
  - Improved the classification step for EEG signals.  
  - Worked on SNGAN fine-tuned for each subject to optimize training time and allow more epochs to be run.  
