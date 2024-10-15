# EEG Classification Project Documentation

## Table of Contents
- [Project Overview](#project-overview)
- [Contributions](#contributions)
- [Expected Outcomes](#expected-outcomes)
- [Future Work](#future-work)
- [Model Architectures](#model-architectures)
- [Milestones Achieved](#milestones-achieved)
- [License](#license)

## Project Overview
This project aims to classify EEG signals using various deep learning architectures, enhancing the understanding of brain activity patterns. We implemented multiple models, including CNN-LSTM hybrids and encoder-based architectures, to achieve improved classification accuracy.

## Contributions
Our project was a collaborative effort with contributions from all five members:
- **Byomakesh Panda**: Worked on the CNN2D-LSTM model and played a key role in the analysis of spatial and temporal feature extraction.
- **Chinmay Bakhale**: Developed the deep CNN architecture and contributed to the comparison of different models' performance.
- **Raghav Borikar**: Focused on the encoder-based models, ensuring efficient feature representation for better EEG classification.
- **Anshal Khatri**: Implemented the CNN1D-LSTM model and contributed to the performance evaluation section of the report.
- **Maneesh Bhaskar**: Assisted with model implementation and analysis.

In addition, all five members contributed equally to writing the report, with sections distributed evenly among the team. The GitHub repository for the project is maintained collectively by the entire group.

## Expected Outcomes
1. **Improved Classification Accuracy**: Our current model achieves ~45% accuracy in classifying EEG signals, and we aim to increase it to 70-80% by refining the CNN-LSTM hybrid model for spatial and temporal pattern recognition.
  
2. **Enhanced Feature Representation**: Temporal, spatial, and residual encoders help capture refined features from EEG signals, leading to a better understanding of brain activity patterns.

3. **Model Performance Comparison**: Through implementing CNN1D-LSTM, CNN2D-LSTM, deep CNN, and encoder-based architectures, we compare and identify the most effective approach, with CNN-LSTM showing promising results.

4. **EEG-based Image Generation**: Leveraging SNGANs, we aim to generate images from classified EEG signals, bridging brain activity and visual stimuli.

5. **Advancement in BCI Technology**: Our study contributes to the future of brain-computer interfaces, enhancing EEG signal decoding and visual stimulus reconstruction.

## Future Work
We plan to focus on the following areas to further improve our project:
- **Increasing Accuracy**: Our target is to enhance the classification accuracy to 75% by refining our models and optimizing hyperparameters.
- **Generating Images**: We will explore advanced techniques to generate images from classified EEG signals, utilizing SNGANs for effective visual representation.

## Model Architectures
### 1. EEG Encoder Model
The EEG Encoder model focuses on processing EEG signals for classification tasks. The architecture consists of various components designed to extract both temporal and spatial features from the EEG data:
- **Input Structure**: Takes a tensor where each row represents an EEG channel, and each column corresponds to a time sample.
- **Temporal Blocks**: Uses multiple 2D temporal convolutional layers with different dilation rates to capture long-range dependencies in the time series.
- **Spatial Features**: Extracts spatial information through spatial convolutional layers to learn patterns across different EEG channels.
- **Output Layer**: Outputs a vector of classes, indicating the predicted class based on the maximum value.

### 2. CNN1D-LSTM Model
This model combines 1D Convolutional Neural Networks (CNN) with Long Short-Term Memory (LSTM) networks to effectively classify time-series data:
- **CNN Layers**: Extract local features from the EEG signals, capturing short-term dependencies.
- **LSTM Layer**: Processes the sequential data output from the CNN layers, learning long-term dependencies and temporal relationships.
- **Fully Connected Layer**: Provides the final classification based on the LSTM's output.

### 3. CNN2D-LSTM Model
This architecture integrates 2D CNNs with LSTM layers to leverage both spatial and temporal features:
- **2D Convolutional Layers**: Capture spatial correlations between EEG channels and time samples.
- **LSTM Layers**: Process the output of the 2D convolutions to model temporal dynamics, enhancing classification accuracy.

### 4. Deep CNN Architecture
A deep CNN model designed to learn hierarchical representations from EEG signals:
- **Multiple Convolutional Layers**: Stack of convolutional layers to extract increasingly complex features from the input data.
- **Pooling Layers**: Reduce dimensionality while preserving essential features, improving model efficiency.
- **Fully Connected Layers**: Classifies the extracted features into predefined classes.

## Milestones Achieved
- **Initial Accuracy**: Achieved ~45% accuracy in classifying EEG signals.
- **Model Development**: Successfully implemented and tested multiple models, including CNN1D-LSTM, CNN2D-LSTM, and encoder-based architectures.
- **Comparative Analysis**: Conducted a thorough comparison of model performances, identifying strengths and weaknesses in different architectures.
- **Image Generation Progress**: Started exploring methods to generate images from classified EEG signals using SNGANs.


