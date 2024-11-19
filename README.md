# EEG Classification and Image Visualization Project Documentation

## Project Overview
This project aims to classify EEG signals using various deep learning architectures, enhancing the understanding of brain activity patterns and reconstructing the image from the EEG signals. We implemented multiple models, including CNN-LSTM hybrids and encoder-based architectures, to achieve improved classification accuracy. For the image generation task we implemented spectrally normalised GAN and conditional GAN models.

## Outcomes
1. **Improved Classification Accuracy**: Our current model achieves an accuracy of 93% in classifying EEG signals which was achieved by redifining and refining the CNN model for spatial and temporal pattern recognition. The EEG signal classification into 40 classes is achieved through various othier models also of which the best performing is the CNN model.
  
3. **Feature Representation**: Using temporal, spatial, and residual blocks of the encoder architecture, captured refined features from EEG signals, leading to a better understanding of brain activity patterns and represented it in a 128 dimensional latent feature vector. This feature vector is used as input for the GAN architecture.

4. **Model Performance Comparison**: Through implementing LSTM-CNN,CNN1D-LSTM, CNN2D-LSTM, deep CNN, and Encoder-based architectures, we compare and identify the most effective approach, with CNN architecture showing promising results. In case of image generation task neither the spectrally normalised GAN nor the conditional GAN could produce conclusive results.

5. **SNGAN and CGAN based Image Generation**: Leveraging SNGAN, we implemented the fetaure vector or the latent feature vector with added noise as input into the generator model, expecting images of 128 x 128 dimensions of the what the visual stimuli was presented to each user.

6. **Approaches for Image Generation**:
    1. We implemented a different approach for the image generation task, where the GAN architecture was trained separately on only one subject from the 6, accounting for different stimulus for each image for             each person based on personal experiences. Hence narrowing down to one subject would increase the chances of patterns in the EEG data for each class of image.
    2. Leveraging the classification task of the EEG data, a ResNet50 model was trained to classify the imaeges into feature vectors for each class, which further acted as input for the GAN archutecture producing         images pertaining to the specific class that was given as stimuli.

8. **Advancement in BCI Technology**: Our study contributes to the future of brain-computer interfaces, enhancing EEG signal decoding and visual stimulus reconstruction.

## Future Work (to be removed)
We plan to focus on the following areas to further improve our project:
- **Increasing Accuracy**: Our target is to enhance the classification accuracy to >80% by refining our models and optimizing hyperparameters.
- **Generating Images**: We will explore advanced techniques to generate images from classified EEG signals, utilizing SNGANs for effective visual representation.
