<a name="br1"></a> 

Image Classification and Text Detection with reCAPTCHA and CAPTCHA

Image Classiﬁcation and Text Detection with reCAPTCHA

and CAPTCHA

Kenny Li

Yale University

kenny.li@yale.edu

Jason Zheng

jason.zheng@yale.edu

Yale University

Abstract

This project explores the application of deep learning techniques for the classiﬁcation and

text detection in CAPTCHA and reCAPTCHA images, two prevalent forms of image-

based security measures. We introduce two distinct neural network architectures tailored

to these challenges. For CAPTCHA images, an Optical Character Recognition (OCR)

model employing a Convolutional Recurrent Neural Network (CRNN) with Long Short-

Term Memory (LSTM) cells and Connectionist Temporal Classiﬁcation (CTC) is presented.

It demonstrated high accuracy, exceeding 90% on diverse datasets, with a notable accu-

racy of over 98% on a smaller dataset (1,070 images) and more than 92% on a larger

one (82,300 images). The approach to reCAPTCHA images involved constructing a deep

Convolutional Neural Network (CNN). We initially employed the Resnet50 architecture,

later transitioning to InceptionV3, both augmented with custom layers for enhanced per-

formance. This adaptation was key in achieving consistent accuracy levels above 80% on a

dataset of 11,730 images. Moreover, the model’s training on an extensive dataset of over

200,000 images demonstrates its scalability and adaptability. These results illustrate the

eﬀectiveness of the proposed models in tackling complex image-based security systems and

contribute to the broader ﬁeld of applying advanced neural network techniques in enhanc-

ing digital security mechanisms.

Keywords: Optical Character Recognition, Convolutional Neural Network, Recurrent

Neural Network, Connectionist Temporal Classiﬁcation Loss, Long Short-Term Memory

Introduction

The advent of the internet brought with it a myriad of opportunities and challenges, one of

which is the distinction between human and automated access. To address this, CAPTCHA

(Completely Automated Public Turing test to tell Computers and Humans Apart) was in-

troduced. Initially patented in 1997, CAPTCHA’s primary function was to prevent auto-

mated bots from exploiting web services. The ﬁrst commercial use of CAPTCHA was the

Gausebeck–Levchin test in 2000, used by idrive.com and PayPal to prevent fraud. The clas-

sic CAPTCHA involved distorted text that bots found challenging to decipher, eﬀectively

blocking their interaction with websites.

Over time, various types of CAPTCHAs evolved, including text-based, image-based, and au-

dio CAPTCHAs, each catering to diﬀerent aspects of bot prevention. Text-based CAPTCHAs,

1



<a name="br2"></a> 

Li and Zheng

the oldest form, presented known words or random warped texts in a visually distorted form.

Image-based CAPTCHAs required users to identify speciﬁc features or elements in a set

of pictures, while audio CAPTCHAs, developed for visually impaired users, played a series

of letters and numbers for the user to input. Despite their eﬀectiveness, these methods

faced challenges with accessibility for visually impaired individuals and were susceptible to

advances in bot capabilities.

The introduction of reCAPTCHA by Google marked a signiﬁcant evolution in this tech-

nology. Developed at Carnegie Mellon University and acquired by Google in 2009, re-

CAPTCHA utilized real-world images and text, such as street addresses and excerpts

from printed books, making it more complex for bots to solve. The initial versions of

reCAPTCHA helped digitize text from sources like The New York Times archives and

Google Books, leveraging human input for a dual purpose: veriﬁcation and digitization.

The NoCAPTCHA reCAPTCHA, a more recent advancement, uses sophisticated methods

like tracking the user’s cursor movements and assessing browser cookies and device history

to determine if a user is a bot.

The eﬀectiveness of CAPTCHA and reCAPTCHA systems as a security measure has been

a subject of debate, given their susceptibility to advanced bots and the potential intrusion

of user privacy. Thus, the motivation behind our project is to develop models that can

accurately classify and detect text in CAPTCHA and reCAPTCHA images, pushing the

boundaries of what is possible in machine learning and artiﬁcial intelligence.

Initially, our motivation was to build an automatic bot that would scrape the CAPTCHA

and reCAPTCHA data live from a website, run it through our machine learning models,

and then auto-populate the correct result, but we quickly realized that this was against

many websites’ terms of service. Thus, we primarily focused on developing the machine

learning models.

For handling text-based CAPTCHAs, the project employs an Optical Character Recog-

nition (OCR) framework, underpinned by a sophisticated Convolutional Recurrent Neural

Network (CRNN). This network integrates a Connectionist Temporal Classiﬁcation (CTC)

output layer, crucial for decoding sequences without explicit segmentation. This conﬁgura-

tion is designed to handle the varying distortions and styles inherent in CAPTCHA texts,

thereby enhancing the model’s adaptability and accuracy.

In tackling the more complex image-based reCAPTCHAs, the project leverages a CNN

that incorporates transfer learning, utilizing Google’s InceptionV3 model. This approach is

selected for its proﬁciency in image analysis and object detection, essential for discerning

the nuanced diﬀerences in reCAPTCHA images.

Initially, the project faced challenges with a large dataset intended for reCAPTCHA train-

ing, particularly due to its size and class imbalance (predominantly ’car’ images). To

mitigate this, we ﬁrst opted for a smaller, more balanced dataset to avoid model bias and

ensure eﬃcient training. Moreover, to stay within the computational constraints of plat-

2



<a name="br3"></a> 

Image Classification and Text Detection with reCAPTCHA and CAPTCHA

forms like Google Colab and Kaggle Notebooks, we were challenged with identifying more

computationally eﬃcient architectures that still produced the desired accuracy.

Adjustments were made to address overﬁtting and runtime eﬃciency. Callback functions

and data augmentation techniques were implemented to facilitate model convergence and

prevent overﬁtting. These modiﬁcations were critical in optimizing the model’s training

within the available computational resources and time constraints.

Data

• Small CAPTCHA Dataset: Sourced from [https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images)

[fournierp/captcha-version-2-images](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images), this dataset provides a collection of 1070

CAPTCHA images. Each image contains a sequence of 5 characters, limited to low-

ercase letters or numbers. They are 200 x 50 grayscale PNGs.

• Large CAPTCHA Dataset: Available at [https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/akashguna/large-captcha-dataset/data)

[akashguna/large-captcha-dataset/data](https://www.kaggle.com/datasets/akashguna/large-captcha-dataset/data), this extensive dataset oﬀers a diverse

range of CAPTCHA images, allowing for the development of more robust text recog-

nition models. Each image again contains a sequence of 5 characters, although the

limitation to lowercase letters or numbers has been relaxed to just numbers and let-

ters. The images are 256 x 256 RGB PNGs, and thus had to be pre-processed before

being input into our model.

• reCAPTCHA Dataset: The dataset for reCAPTCHA, found at [https://www.](https://www.kaggle.com/datasets/mikhailma/test-dataset)

[kaggle.com/datasets/mikhailma/test-dataset](https://www.kaggle.com/datasets/mikhailma/test-dataset), contains 11,730 images across 12

classes. These images are representative of those used in Google’s reCAPTCHA V2,

ideal for training image-based recognition models.

Methodology

CAPTCHA OCR

A CRNN model integrating LSTM cells and a CTC layer was developed, following the archi-

tecture in [Shi](#br7)[ ](#br7)[et](#br7)[ ](#br7)[al.](#br7)[ ](#br7)[(2017).](#br7)[ ](#br7)The model initiates with convolutional layers (Conv2D) using

SELU activation and same padding for primary feature extraction from images. This is fol-

lowed by Max pooling (MaxPool2D) to reduce feature dimensionality. Batch normalization

(BatchNormalization) is included after speciﬁc convolutional layers to enhance stabilization

and training eﬃciency. For sequence processing, the model utilizes bidirectional LSTM

layers (Bidirectional, CuDNNLSTM), critical for capturing sequential dependencies within

character series. The architecture progresses to a Map-to-Sequence stage and then to a

dense layer (Dense) with SELU activation, facilitating the transition from feature maps

to sequence predictions. This is followed by a second batch normalization layer for fur-

ther stabilization. The ﬁnal classiﬁcation of characters is performed by a softmax layer

(Dense with softmax activation), determining the probability distribution over the charac-

ter set. The model employs the Adam optimizer for eﬀective training. A custom CTC loss

layer (CTCLayer) is integrated, crucial for sequence alignment; it computes the probability

of the target sequence given the model’s inputs, enabling the handling of variable-length

3



<a name="br4"></a> 

Li and Zheng

sequences and alignment between predictions and ground-truth labels without predeﬁned

segmentation, a concept inspired by [Graves](#br7)[ ](#br7)[et](#br7)[ ](#br7)[al.](#br7)[ ](#br7)[(2006).](#br7)

Layer Type

Input

Conv2D

Speciﬁcations

Layer Name

Image

Conv1

Shape: (256, 80, 1) (transposed)

64 ﬁlters, 3x3, SELU, same padding

2x2, stride 2

MaxPool2D

Conv2D

MaxPool1

128 ﬁlters, 3x3, SELU, same padding Conv2

2x2, stride 2

MaxPool2D

Conv2D

Conv2D

MaxPool2

256 ﬁlters, 3x3, SELU, same padding Conv3

256 ﬁlters, 3x3, SELU, same padding Conv4

MaxPool2D

Conv2D

2x1, stride 2

512 ﬁlters, 3x3, SELU, same padding Conv5

MaxPool3

BatchNormalization

Conv2D

BatchNormalization1

512 ﬁlters, 3x3, SELU, same padding Conv6

BatchNormalization2

MaxPool4

BatchNormalization

MaxPool2D

2x1, stride 2

Map-to-Sequence and Sequence Processing Begins

512 ﬁlters, 2x2, SELU, valid padding Conv7

Conv2D

Reshape

Squeeze to new dimensions

512 units, SELU

Reshape

Dense1

Dense

BatchNormalization

BatchNormalization3

Bi-LSTM1

Bi-LSTM2

Dense2

Bidirectional LSTM 256 units, return sequences

Bidirectional LSTM 256 units, return sequences

Dense

CTCLayer

Softmax activation

CTC Loss

Table 1: Layers of the CRNN Model

reCAPTCHA CNN

For the reCAPTCHA model, we decided to build and implement a Convolutional Neural

Network. We built a class, InceptionV3Model, that took advantage of the pre-existing In-

ceptionV3 model within Keras. This model was built based oﬀ of Szegedy et al’s ’Rethinking

the inception architecture for computer vision’. Additionally, we added a few custom layers

on top of the pre-existing architecture of the Inceptionv3 model. These custom layers in-

cluded a GlobalAveragePooling2D layer, 2 Dropout layers, and 2 Dense layers. To further

enhance our model, we unfreeze more layers of the Inceptionv3 model after 10 epochs are

ran. The optimizer used in our model was Stochastic Gradient Descent.

4



<a name="br5"></a> 

Image Classification and Text Detection with reCAPTCHA and CAPTCHA

Layer Type

InceptionV3

GlobalAveragePooling2D

Dropout

Speciﬁcations

Layer Name

Input Shape: (150, 150, 3) BaseModel

GlobalPool

Rate: 0.7

Dropout1

Dense1

Dense

Dropout

Dense

256 Units, RELU

Rate: 0.5

Softmax Activation

Dropout2

Dense2

Table 2: Layers of the CNN Model

Implementation Details

CAPTCHA OCR

For the CAPTCHA OCR, a CRNN model was implemented using TensorFlow and Keras,

with speciﬁc conﬁgurations for training and preprocessing. The key steps in the implemen-

tation are as follows:

1\. Hyperparameters and Callbacks: The model was trained with a batch size of 128,

and an Adam optimizer was used with an initial learning rate of 0.001, beta1 set to

0\.9, beta2 set to 0.999, and a clip norm of 1.0. Training was conducted for a maximum

of 100 epochs. To prevent overﬁtting and ensure eﬃcient training, an early stopping

callback with a patience of 10 epochs and a ReduceLROnPlateau callback to adjust

the learning rate based on the validation loss were employed. ReduceLROnPlateau

monitored the val loss and reduced the learning rate by a factor of 0.1 to a minimum

of 1 × 10<sup>−5</sup> with a patience of 5 epochs.

2\. Image Preprocessing: The images were resized to ﬁt the input requirements of the

model, with a width of 256 and a height of 80 pixels. This preprocessing involved

either cropping or padding the images to maintain their aspect ratio. Additionally,

images were converted to grayscale and their data type was changed to ﬂoat32. Prior

to being input into the image, the pixels were also transposed.

3\. Dataset Preparation: The dataset was split into training and testing sets with a

ratio of 80:20. For each image in the dataset, the encode single sample function was

applied, which handled the image resizing and label encoding. Labels were encoded

using a StringLookup layer, converting each character in the label to its corresponding

integer representation.

4\. Data Augmentation: The training dataset was augmented to include a variety of

transformations to improve the robustness of the model. The augmentation included

random adjustments to the image brightness, contrast, and orientation.

5\. Model Training: The training and validation datasets were fed into the CRNN

model for training. The custom CTCLayer calculated the CTC loss between the

predicted output and the true labels during training.

6\. Model Evaluation: Post-training, the model’s performance was evaluated on the

test dataset to assess its accuracy and generalization capabilities.

5



<a name="br6"></a> 

Li and Zheng

reCAPTCHA CNN

The reCAPTCHA CNN model was also implemented using TensorFlow and Keras, with

speciﬁc conﬁgurations for training and preprocessing. The key steps in the implementations

are as follows:

1\. Hyperparameters and Callbacks: The model was trained with a batch size of 64,

and a Stochastic Gradient Descent optimizer was used with a learning rate of 0.001.

Training was conducted for a maximum of 100 epochs. To prevent overﬁtting and

ensure eﬃcient training, an early stopping callback with a patience of 10 epochs and

a ReduceLROnPlateau callback to adjust the learning rate based on the validation

loss were employed. After 10 epochs, the model is stopped and unfreezes some more

layers, to allow for more ﬁnite tuning. The model is then restarted on the remainder

of the 100 epochs with a new learning rate of 0.0001.

2\. Image Preprocessing: The images were resized to ﬁt the input requirements of the

model, with a width of 150 and a height of 150 pixels.

3\. Dataset Preparation: The dataset was split into training and validation sets with

a ratio of 80:20.

4\. Data Augmentation: The training dataset was augmented to include a variety of

transformations to improve the robustness of the model. The augmentation included

random adjustments to the image brightness, contrast, and orientation via random

rotations, zooms, and shifts.

5\. Model Training: The training and validation datasets were fed into the CNN model

for training.

6\. Model Evaluation: Post-training, the model’s performance was evaluated on the

validation dataset to assess its accuracy and generalization capabilities.

Results

We successfully met our primary goal of achieving an accuracy rate of 75% or higher across

all models and datasets.

For the large CAPTCHA dataset, our CAPTCHA OCR model demonstrated a remark-

able accuracy of 95.53% on the test data. This level of accuracy is particularly noteworthy

given the inherent diﬃculty in diﬀerentiating between visually similar characters such as

”I”, ”1”, ”7”, or ”o”, ”O”, ”0”. The model’s ability to discern these subtle diﬀerences,

which can even challenge human perception, underscores its robustness and precision in

handling complex CAPTCHA images.

On the small CAPTCHA dataset, the model achieved an even higher accuracy of 98.13%

on the test data, further solidifying the eﬀectiveness of our approach in dealing with varied

CAPTCHA challenges, albeit on a much smaller dataset.

6



<a name="br7"></a> 

Image Classification and Text Detection with reCAPTCHA and CAPTCHA

The reCAPTCHA model, designed to tackle more complex image-based recognition tasks,

attained a commendable accuracy of 81.60% on the validation dataset. This performance is

particularly signiﬁcant considering the diverse and intricate nature of reCAPTCHA images,

which are designed to be challenging for automated systems.

References

A. Graves, S. Fern´andez F. Gomez, and J. Schmidhuber. Connectionist temporal classiﬁ-

cation: Labelling unsegmented sequence data with recurrent neural networks. In ICML

2006 - Proceedings of the 23rd International Conference on Machine Learning, pages

369–376. ACM, 2006.

B. Shi, X. Bai, and C. Yao. Approximating discrete probability distributions with depen-

dence trees. IEEE Transactions on Information Theory, 39:462–467, 2017.

C. Szegedy, V. Vanhoucke, and J. Shlens. Rethinking the inception architecture for com-

puter vision. 2016.

[Szegedy](#br7)[ ](#br7)[et](#br7)[ ](#br7)[al.](#br7)[ ](#br7)[(2016)](#br7)[ ](#br7)[Shi](#br7)[ ](#br7)[et](#br7)[ ](#br7)[al.](#br7)[ ](#br7)[(2017)](#br7)[ ](#br7)[Graves](#br7)[ ](#br7)[et](#br7)[ ](#br7)[al.](#br7)[ ](#br7)[(2006)](#br7)

7


