# Image Classification and Text Detection with reCAPTCHA and CAPTCHA
=

## Introduction
This project explores deep learning techniques for classifying and detecting text in CAPTCHA and reCAPTCHA images, introducing two neural network architectures tailored for these challenges: an OCR model for CAPTCHA images and a CNN model for reCAPTCHA images. The OCR model demonstrated high accuracy, exceeding 90% on diverse datasets. For reCAPTCHA images, using architectures like Resnet50 and later InceptionV3, we achieved consistent accuracy levels above 80%.

## Methodology
### CAPTCHA OCR
We employed a Convolutional Recurrent Neural Network (CRNN) model integrating Long Short-Term Memory (LSTM) cells and Connectionist Temporal Classification (CTC) for CAPTCHA image text detection. This model effectively handles the distortions and styles inherent in CAPTCHA texts.

### reCAPTCHA CNN
For image-based reCAPTCHA challenges, we leveraged a CNN that incorporates transfer learning, utilizing the InceptionV3 model. This approach is selected for its proficiency in image analysis and object detection, essential for discerning nuanced differences in reCAPTCHA images.

## Data
We used three datasets for our experiments:
- Small CAPTCHA Dataset from Kaggle for basic CAPTCHA images.
- Large CAPTCHA Dataset from Kaggle for a diverse range of CAPTCHA images.
- reCAPTCHA Dataset from Kaggle for training image-based recognition models on Google’s reCAPTCHA V2 images.

## Results
Our models achieved significant accuracy rates:
- CAPTCHA OCR model: Over 98% on a smaller dataset and more than 92% on a larger one.
- reCAPTCHA CNN model: Consistent accuracy levels above 80% on a dataset of 11,730 images, demonstrating scalability and adaptability.

## References
- A. Graves et al., "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," ICML 2006.
- B. Shi et al., "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.
- C. Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," 2016.

## License
[Specify the license here]

