# Detecting-Deceptive-Deepfake-Media-Using-Generative-Adversarial-Networks-


Technological advancements have revolutionized digital communication, resulting in the widespread use of social media for sharing opinions and information. However, this surge in digital interaction has also facilitated the spread of fake news and rumours, with deepfake technology emerging as a significant concern. Deepfakes, which use Generative Adversarial Networks (GANs) to create highly realistic synthetic media, especially through face-swapping in videos, can manipulate individuals' words and actions, spread false information, and damage reputations. This research aims to address these challenges by exploring the use of GANs to detect deepfakes.
The study has three primary goals: to identify the most effective architectures and techniques for detecting deepfakes, to analyse how different types of deepfakes impact GAN-based detection methods, and to develop a GAN model specifically designed to recognize deceptive deepfake content. The methodology involves compiling a diverse dataset of real and deepfake images/videos, testing different GAN architectures such as Basic GAN, Deep Convolutional GAN (DCGAN), StyleGAN, and XceptionNet, and training these models to distinguish between genuine and fake media. The evaluation will be based on accuracy metrics such as recall, precision, ROC-AUC and F1-score. The research is implemented using Jupyter Notebook, focusing on practical demonstration and analysis. By leveraging GANs' capabilities for pattern recognition and data synthesis, the study aims to provide a robust, scalable solution to the growing issue of deepfake media, enhancing the integrity of digital information and ensuring media authenticity.


Research Design

Image/Video Model Implementation

The research design for deepfake detection followed a structured approach, focusing on building a Convolutional Neural Network (CNN) using the Xception architecture to classify images as real or fake. The overall process involved four key stages: data collection, preprocessing, model development, and performance evaluation.

Data Collection and Preprocessing

The beginning of our research involved collecting a labelled data of real and fake images. These images were organized into two directories, one for each category: real images and fake images. Preprocessing steps were applied to ensure uniformity across the dataset before feeding it into the model. This included:
•	Loading and resizing: Each image was loaded using the load_img function and resized to a consistent target size of 128x128 pixels to standardize input dimensions.
•	Converting images to arrays: The images were transformed into NumPy arrays using the img_to_array function, allowing them to be compatible with the model.
•	Normalizing pixel values: Image pixel values were scaled to a ratio [0, 1], helping the neural network train more efficiently by speeding up convergence.
Finally, each image was assigned a label (0 for real and 1 for fake) based on its directory, which facilitated the binary classification task.
Datasets (Links): 
Deep Fake Voice Recognition (kaggle.com)
140k Real and Fake Faces (kaggle.com)

Dataset Partitioning

After preprocessing, dataset is split in two: training and validation parts. Training set contained 80% of the data, while 20% was reserved for validation. This split was performed using the train_test_split function, which shuffled the dataset to create distinct training and validation subsets. The purpose of the validation set was to evaluate the model’s ability to generalize and avoid overfitting.
An additional test set composed of only fake images was pre-processed similarly. This test set provided a way to assess the model’s accuracy in detecting fake images in real-world conditions, ensuring that the model could generalize beyond the initial training data.



Model Architecture

The CNN model was built using the Xception architecture, which is known for its superior performance in image classification tasks due to its ability to capture intricate spatial features. The Xception model was pre-trained on the ImageNet dataset, but its fully connected layers were removed to allow for transfer learning, adapting the model for the specific task of deepfake detection.
Key architectural components:
•	Xception as feature extractor: The pre-trained Xception layers acted as the core of the model, extracting detailed features from the images.
•	GlobalAveragePooling2D: This layer reduced the dimensionality of the feature maps while retaining their key features, creating a more efficient model by summarizing the extracted features.
•	Dense layer with sigmoid activation: A fully connected layer with a sigmoid activation function was added as the final layer to output a binary classification (real or fake). The sigmoid function was used because it is well-suited for binary classification tasks, outputting a probability between 0 and 1.

Model Training and Optimization

The model was compiled using the Adam optimizer with a learning rate of 1e-4. Adam was chosen for its ability to dynamically adjust learning rates, promoting faster convergence. The binary cross-entropy loss function was employed to measure the error in binary classification tasks.
•	Training: The model was trained for 10 epochs with a batch size of 32. During each epoch, the model adjusted its weights based on the gradient of the loss function, iteratively improving its ability to classify images.
•	Evaluation: Throughout the training process, the model's performance was evaluated on both the training and validation sets. Metrics like accuracy and loss were recorded at each epoch to monitor the model’s learning progress. This evaluation helped detect issues such as overfitting (when the model performs well on the training data but poorly on unseen data) or underfitting (when the model struggles to learn from the training data).
•	The validation set provided crucial feedback, ensuring that the model was learning to generalize well to new data rather than merely memorizing the training images. By the end of training, the model could effectively distinguish between real and fake content, demonstrating its ability to handle the deepfake detection task.
This approach, cantered on CNN and the Xception architecture, offers a robust method for classifying deepfake content by leveraging both powerful feature extraction and transfer learning, ensuring high accuracy in distinguishing real from fake media.


Evaluation of Testing Data

After training the model, it was evaluated using a separate test dataset containing only fake images. The primary goal of this evaluation was to determine how well the model could classify fake images, assessing its practical ability to detect deepfakes. During the testing phase, predictions were generated using the trained model, and each prediction was compared against the ground truth label (which was 1 for all images in this dataset). The model’s testing accuracy was calculated as the ratio of correctly classified images to the total number of test samples, offering an insight into how well the model generalizes to new, unseen fake content. This testing phase was crucial in determining whether the model could effectively identify deepfakes outside of the controlled training environment.

Performance Visualization

To further understand the model's behaviour during training, performance metrics such as accuracy and loss were visualized over time using plots. These visualizations provided a clearer picture of the model’s learning process by highlighting:
Training accuracy/loss: Showing how well the model performed on the training data over time.
Validation accuracy/loss: Providing a measure of the model’s generalization ability to unseen data.
These plots helped identify whether the model was converging as expected or experiencing issues like overfitting (performing well on training data but poorly on validation data). By examining the trends in accuracy and loss over the epochs, it was possible to evaluate whether adjustments to parameters such as learning rate or training epochs were needed.

Model Deployment

Once training and evaluation were complete, the final model was saved in Keras format, making it ready for deployment in real-world applications. This step was essential for transferring the model to a production environment, where it could be used to detect deepfake content in multimedia. The saved model could be reloaded in future sessions for further fine-tuning or integrated into larger systems designed to detect deepfake videos and images in real-time settings.

Audio Model Implementation

In parallel with the image model, a deepfake audio detection model was also developed using a systematic approach that included data preprocessing, feature extraction, model design, training, and evaluation. The objective of the audio model was to classify audio samples as either "real" or "fake" using a 1-dimensional Convolutional Neural Network (CNN) tailored for binary classification.
Audio Feature Extraction
The first step in developing the audio model was extracting relevant features from the audio data. Audio samples were collected from real and fake sources and processed using the librosa library. The primary feature extracted was the Mel-Frequency Cepstral Coefficients (MFCCs), a popular representation of the power spectrum in audio and speech analysis. To ensure uniformity and consistency, the MFCCs were averaged over time for each coefficient, resulting in a fixed-size feature vector for each audio file. This compact representation captured the essential frequency and temporal characteristics needed for accurate deepfake voice detection.

Dataset Preparation and Labelling

The audio dataset was categorized into two classes: "real" and "fake." Corresponding MFCC feature vectors were labeled, with "real" samples encoded as 0 and "fake" samples encoded as 1. The dataset was then split into training and testing sets using an 80-20 split. This split ensured that the model was trained on a portion of the data and evaluated on unseen data, providing an unbiased measure of performance. Proper labelling and partitioning were crucial for assessing the model’s ability to generalize and detect fake audio effectively.

Data Reshaping for CNN Input

Before feeding the data into the CNN, the feature vectors are reshaped to be compatible with the input requirements of the model. Since the CNN expects a three-dimensional input, the feature vectors (MFCCs) were expanded along a new axis to create a structure suitable for 1D convolution. This reshaping allowed the CNN to effectively learn patterns from the temporal relationships within the audio features.

Model Architecture

The deepfake audio detection model is designed using a 1D Convolutional Neural Network (CNN), which is effective for analyzing sequential data like audio signals. The architecture includes:
Two convolutional layers, each followed by max-pooling and dropout layers. The first convolutional layer applies 64 filters with a kernel size of 3 to capture local patterns in the sequential data.
-	Max-pooling is used to down-sample the input, reducing its dimensionality while retaining essential features.
-	A dropout layer with a rate of 30% helps mitigate overfitting by randomly deactivating a portion of neurons during each training step.
The second convolutional layer applies 128 filters, which capture more complex patterns, followed by another round of max-pooling and dropout.
After the convolutional layers, the output is flattened into a one-dimensional vector.
The flattened vector is passed through a fully connected (dense) layer with 64 units, activated using the ReLU function, which introduces non-linearity.
Finally, a sigmoid activation function is applied in the output layer for binary classification (real or fake), outputting a probability between 0 and 1.

Model Training

The model was trained with the Adam optimizer, recognized for its effectiveness in deep learning applications, and employed binary cross-entropy as the loss function, a common choice for binary classification tasks. Key training parameters included:
•	50 epochs: The model was iteratively trained for 50 cycles to optimize its weights.
•	Batch size of 32: Each epoch processed batches of 32 audio samples at a time.
•	A 20% validation split: Performance monitored on unseen data and to avoid overfitting.
Throughout the training process, accuracy and loss metrics were tracked for both the training and validation datasets.

Model Evaluation

After training, the model was tested on a separate dataset to assess its performance. The evaluation phase involved:
•	Computing the final accuracy on the test set, indicating the proportion of correctly classified audio samples.
•	Generating a classification report, which provided more detailed metrics such as:
•	Precision: The proportion of true positives among the predicted positives.
•	Recall: The proportion of true positives identified out of all actual positives.
•	F1-score: The harmonic mean of precision and recall, providing a balance between them.
•	Visualizing the model's learning process by plotting the training and validation accuracy and loss over the epochs. These plots helped identify trends in the model’s learning, ensuring that it was generalizing well and not overfitting to the training data.

Model Deployment

Once the model training is complete, it is saved in the keras format for future use in real-time applications. This final step ensured that the model could be deployed in practical scenarios, enabling it to perform deepfake detection on new, unseen audio data.

Both the image and audio models relied on preprocessing, feature extraction, and state-of-the-art neural network architectures to effectively detect deepfake content across multiple media formats. These models represent powerful tools for combating the growing threat of deepfake manipulation in digital content.

Dashboard Implementation

The implementation of a deepfake detection dashboard involved a structured research design to integrate machine learning models into a user-friendly interface using Streamlit. The primary objective was to detect deepfakes across three media types: images, videos, and audio. Critical to the system functionality is the loading of pre-trained deep learning TensorFlow models. These models are essential for processing media inputs and predicting whether the content is real or manipulated. To ensure computational efficiency, both the image/video model and the voice detection model were cached using Streamlit’s @st.cache_resource decorator, which optimizes the application by preventing repeated reloading during the session.

With various medias being available within the public domain, the deepfake detection mechanisms were tailored for majority popular media type. Therefore, requiring different preprocessing techniques. For image detection, the system resizes uploaded images to a standard dimension of 128x128 pixels and normalized the pixel values. The processed image are then passed through the detection model, which produce a confidence score to classify the image as either "Fake" or "Real." Video detection followed a more complex approach, where the video was read frame by frame. Every tenth frame is extracted, resized, and converted to an array for batch prediction. The overall classification is determined by averaging the prediction scores of the frames. For audio detection, the system extracted Mel-Frequency Cepstral Coefficients (MFCCs) from the audio file, which served as input features for the deepfake model. The MFCC features were reshaped to match the model’s input format, and the prediction result indicated whether the audio was genuine or synthetic.

The dashboard interface is designed to be intuitive. Users can upload media files for analysis. The file uploader supports a range of formats, including images (JPEG, PNG), videos (MP4), and audio (WAV, MP3). The system automatically detects the media type and routes it to the appropriate deepfake detection function. The detection results are presented clearly: uploaded images are displayed along with their deepfake classification and confidence score. Videos and audio files are rendered for playback before the analysis results were shown.

To manage media files during the analysis, temporary storage is implemented for videos and audio files. This approach allows the system to save the uploaded media temporarily for processing. Files are then deleted after the analysis is completed, ensuring efficient use of resources. The dashboard also features an informative sidebar, offering users an explanation of the tool’s purpose and the machine learning technology behind it. This section is aimed to enhance user understanding of how the deepfake detection models operate and the importance of identifying manipulated media.

In summary, the deepfake detection dashboard combined sophisticated machine learning models with a streamlined interface to provide real-time analysis of images, videos, and audio. By leveraging TensorFlow and Streamlit, the system offers efficient performance and user-friendly functionality, making it a valuable tool for detecting media manipulation. The design emphasizes both technical accuracy and ease of use, ensuring that users can confidently identify deepfakes with minimal input.

Data Collection Methods

Data collection for this research was conducted using publicly available datasets with appropriate Creative Commons licenses (CCL), allowing for ethical use in deepfake detection development. Various image and video frame datasets were sourced from open repositories that specialize in manipulated media, such as deepfake-specific datasets and general facial recognition datasets containing both real and synthetic content. These datasets provided a diverse range of authentic and tampered images and video frames. Ultimately enabling robust training and testing of the machine learning models. Audio datasets were also sourced, including both genuine and synthetically generated voice samples. This allowed for the integration of voice deepfake detection into the system, broadening the scope of the research. The use of licensed datasets ensured compliance with data usage rights while facilitating the acquisition of a wide array of data for comprehensive model training and evaluation.

Data Analysis Techniques

Image/Video Analysis Techniques
The analysis techniques used in developing the deepfake detection model were aimed at ensuring that the model could accurately classify images/videos as either real or fake. These techniques were employed to process and analyse the dataset, optimize model performance, and evaluate its classification abilities. The following sections outline the key methods used for data analysis and model evaluation.

Data Preprocessing and Normalization

The first stage of analysis involved preprocessing the image data to prepare it for model training. Images were loaded using the load_img and img_to_array functions, where each image was resized to 128x128 pixels to ensure uniformity across the dataset. A critical analysis step was the normalization of pixel values, where all images were scaled to the range [0, 1]. This normalization process is an essential technique in deep learning, as it helps prevent issues related to varying pixel intensities and ensures faster convergence of the model during training. By scaling the pixel values, the model was able to process the input more efficiently, reducing the chances of overfitting or poor generalization.

Label Encoding and Dataset Partitioning

To facilitate binary classification, the real and fake images were labelled numerically: 0 for real and 1 for fake. This enabled the model to output predictions for discrete classes during training. The dataset was then split into training and validation sets using an 80-20 ratio. The train_test_split function was used to ensure that both sets had a balanced distribution of real and fake images, which is important for obtaining reliable feedback on the model’s performance across both training and unseen validation data.
The validation set was crucial for assessing overfitting, where the model might perform well on training data but fail to generalize to new examples. By comparing training accuracy with validation accuracy at each epoch, the researchers could determine whether the model was learning effectively or overfitting.

Transfer Learning and Model Design

Transfer learning was employed by leveraging the pre-trained Xception model, which had already been trained on the ImageNet dataset. This model served as a feature extractor, transferring the knowledge it had learned from general image classification to the specific task of deepfake detection.
A GlobalAveragePooling2D layer was used to reduce the dimensionality of the extracted features and prevent overfitting by minimizing the number of parameters in the fully connected layers. This pooling method helped the model generalize by focusing on global rather than localized features.
The final layer was a dense layer with a sigmoid activation function, producing probabilities for the binary classification task (real or fake). The binary cross-entropy loss function was used because it is ideal for binary classification tasks, and the Adam optimizer with a learning rate of 1e-4 was chosen to ensure smooth and stable convergence.

Training and Validation Analysis

During training, the model's performance was tracked through accuracy and loss metrics, computed for each epoch on both the training and validation sets. The loss function measured how well the predicted probabilities matched the true labels, while accuracy quantified how many predictions were correct.
Monitoring these metrics helped detect overfitting. If the training accuracy increased but the validation accuracy plateaued or decreased, this indicated overfitting. To address this, the training process was regularly monitored, and adjustments were made, such as reducing the number of epochs or increasing regularization (e.g., through dropout layers). Dropout was used to randomly drop units during training, forcing the model to generalize better by avoiding reliance on specific neurons.

Testing Analysis on Unseen Data

Once training was complete, the model was evaluated on a test dataset composed solely of fake images. This allowed for an assessment of the model's ability to correctly classify fake images in a real-world scenario. The model outputted a probability score for each test image, and a threshold of 0.5 was applied to classify each image as fake or real.
The predictions were compared with the ground truth labels (1 for all test images), and the number of correct classifications was calculated. The testing accuracy was determined by the ratio of correct classifications to the total number of test images, offering insight into how well the model performed outside the training environment.
This testing phase provided a realistic assessment of the model's utility in detecting deepfake images, particularly in situations where the model would encounter unseen data.

Visualization of Training Progress

To further analyse the model’s training process, the accuracy and loss for both training and validation data were plotted over time. These visualizations provided a clear understanding of how the model’s learning evolved with each epoch. In particular, it was possible to visually inspect whether the training loss decreased consistently and whether the validation loss exhibited a similar trend, which would indicate that the model was learning generalizable features rather than memorizing the training data.
Such plots also helped identify the point at which training could be halted (early stopping), as further training beyond a certain point might not yield significant improvements. Visualization techniques like this are essential for diagnosing issues like overfitting or underfitting and for guiding decisions about model adjustments.

Performance Metrics

In addition to accuracy, the study relied on other performance metrics such as precision, recall, and F1-score, which could be derived from the confusion matrix when evaluating the model. Although not explicitly mentioned in the code, these metrics provide deeper insight into how well the model is performing, especially in cases of imbalanced datasets where accuracy alone might not provide a full picture. For instance, precision and recall would help measure how well the model is at minimizing false positives and false negatives, respectively, while the F1-score gives a balanced measure of both.

Audio Analysis Techniques

The analysis techniques and methods employed in the creation of the deepfake voice detection model focused on extracting meaningful features from audio data, preprocessing the data for compatibility with machine learning algorithms, and using performance metrics to evaluate the model’s effectiveness. These techniques were designed to maximize the model's ability to distinguish between real and fake audio samples while ensuring robust evaluation and validation of the results.

Feature Extraction Using MFCCs

The central analysis technique used to process audio data was the extraction of Mel-Frequency Cepstral Coefficients (MFCCs). MFCCs are widely regarded as one of the most effective representations of the power spectrum in audio analysis, particularly for speech processing and audio classification tasks. In this model, MFCCs were calculated from each audio file using the librosa library. A total of 40 MFCCs were extracted for each sample, capturing both temporal and spectral characteristics of the audio signal. To standardize the input size for the machine learning model, the mean of each MFCC feature was taken across time, reducing the variability in sequence length and providing a fixed-length feature vector. This dimensionality reduction technique ensured that the model focused on the most critical acoustic information.

Label Encoding and Dataset Preparation

The dataset is labelled into two classes: "real" and "fake." These categorical labels were then encoded into numerical format using LabelEncoder, where the real class was represented by 0 and the fake class by 1. Encoding categorical labels into numeric values is a common preprocessing step in machine learning, as many models, including neural networks, require numerical input to perform computations. Furthermore, the dataset was split into training and testing sets using an 80-20 ratio via train_test_split, a widely used technique for ensuring that the model is trained on most of the data but tested on an independent subset. This split allows for an unbiased evaluation of the model's generalization capability on unseen data.

Data Reshaping for CNN Input

To prepare the extracted MFCC features for input into the Convolutional Neural Network (CNN), the feature vectors were reshaped into a format compatible with 1D convolutions. This involved expanding the feature vectors along a new axis, effectively transforming the 2D MFCC array into a 3D format, where each MFCC sequence was treated as a separate channel. This reshaping is critical for enabling the CNN to capture temporal patterns within the audio features, as the 1D convolutional filters scan over the sequence to identify key acoustic cues associated with real or fake speech.

Model Training and Validation

During the training process, the dataset was split into training and validation sets, with 20% of the data reserved for validation. This allowed the model to be evaluated on unseen data during training, providing real-time feedback on its ability to generalize.
The model was optimized using the Adam optimizer, which adapts the learning rate based on the gradients of the loss function, ensuring stable and efficient convergence. The binary cross-entropy loss function was used, which is specifically tailored for binary classification tasks, measuring the error between the predicted probabilities and the true labels (real or fake).
The training process was tracked over 50 epochs, with accuracy and loss recorded for both the training and validation sets. This tracking was crucial for understanding the model’s learning behavior. If the validation accuracy plateaued or decreased while training accuracy increased, this would indicate overfitting. Conversely, if both accuracies increased steadily, it suggested that the model was effectively learning from the data.

Model Evaluation Techniques

Upon completion of training, the model was evaluated on a test dataset that was kept separate throughout the training process. This provided an unbiased estimate of the model's real-world performance. The evaluation metrics included:
•	Accuracy: The percentage of correct predictions made by the model on the test set, indicating how well the model classified images as real or fake.
•	Classification Report: A comprehensive report that included additional performance metrics such as:
-	Precision: The proportion of correctly predicted positive instances (e.g., fake images) among all predicted positive instances.
-	Recall: The proportion of actual positive instances that were correctly identified by the model.
-	F1-Score: The harmonic mean of precision and recall, offering a balanced evaluation of the model’s ability to detect both real and fake images.

Visualization of Training Progress

To gain further insights into the model’s training dynamics, accuracy and loss curves were plotted for both the training and validation sets across the 50 epochs. These visualizations were instrumental in detecting overfitting and underfitting.
If the training accuracy improved while the validation accuracy plateaued or dropped, it would signal that the model was overfitting to the training data, learning patterns specific to the training set but failing to generalize.
If both accuracies increased steadily, this indicated that the model was effectively learning and generalizing to new data.
Similarly, loss curves were analysed to ensure that the model was minimizing errors over time. A significant gap between training and validation loss would also suggest overfitting. In contrast, a consistent reduction in both training and validation loss would indicate proper learning and model optimization.
These plots provided a clear overview of the model’s performance during the training process, guiding decisions on potential adjustments such as modifying the number of epochs, tweaking regularization parameters, or fine-tuning the learning rate for better generalization. 

=================================================================================

Results

Presentation of Data

<img width="940" height="365" alt="image" src="https://github.com/user-attachments/assets/f5432da9-fbaf-46d5-9186-f591c602ffed" />
Model graph representation

Image/Video Deepfake Detection Representation
<img width="940" height="412" alt="image" src="https://github.com/user-attachments/assets/dd5d9256-3eb8-48ac-8584-c84aaf9a23f6" />

Deepfake detection representation image fake

<img width="940" height="420" alt="image" src="https://github.com/user-attachments/assets/4425722d-f4d0-48ea-b258-ad728168d400" />
 
Deepfake detection representation image real(2)

<img width="940" height="413" alt="image" src="https://github.com/user-attachments/assets/a8c3d4ad-8dbc-4f43-94c3-152d33078121" />

Deepfake detection representation video

<img width="940" height="418" alt="image" src="https://github.com/user-attachments/assets/6c7115f9-e7ae-4722-8c7c-027be4bdc1bd" />

Deepfake detection representation image video(2)


Audio Deepfake Detection Model Evaluation Matrix
	            Precision	Recall	F1-Score	Support
0	            0.77	    0.91	  0.83	    11
1	            0.90	    0.75	  0.82	    12
Accuracy			                  0.83	    23
Macro avg	    0.83	    0.83	  0.83	    23
Weighted avg	0.84	    0.83	  0.83	    23


Audio Deepfake Detection Model Evaluation Matrix Graph Representation
<img width="933" height="350" alt="image" src="https://github.com/user-attachments/assets/3eee997b-0823-4d3a-bb84-6ba07e147af7" />
 
Model graph representation audio

Real Audio Analysis

<img width="919" height="318" alt="image" src="https://github.com/user-attachments/assets/b3133cca-3cfd-489a-b07d-febd3f0f5bb4" />

Real audio wave length

<img width="963" height="389" alt="image" src="https://github.com/user-attachments/assets/dec25f7d-ba7a-4e71-8587-ebc5d18f04ef" />

Real audio spectogram

<img width="926" height="382" alt="image" src="https://github.com/user-attachments/assets/2452efb4-897b-4d2d-abbb-fcd779efaa7b" />

Real audio mel spectrogram

Fake Audio Analysis

<img width="933" height="372" alt="image" src="https://github.com/user-attachments/assets/29be2d68-b091-4d83-89ab-feb70e72658f" />

Fake audio wave length

<img width="930" height="394" alt="image" src="https://github.com/user-attachments/assets/c04c0f02-5735-42b3-b6f5-58345325b060" />

Fake audio spectogram

<img width="930" height="386" alt="image" src="https://github.com/user-attachments/assets/18276b5a-0ff0-4396-9d97-de389a6000ea" /> 

Fake audio mel spectrogram

Audio Deepfake Detection Representation

<img width="942" height="415" alt="image" src="https://github.com/user-attachments/assets/1f08969d-4e72-4ef2-93f4-0b805004eef8" /> 

Deepfake detection representation audio(fake)

<img width="940" height="416" alt="image" src="https://github.com/user-attachments/assets/0d4d5e27-851f-4731-943d-9b86a89e20f6" />

Deepfake detection representation audio(real)

Convolutional Neural Network Architecture 

<img width="928" height="374" alt="image" src="https://github.com/user-attachments/assets/8fc53253-8e1d-4daf-a19e-6bb136041931" /> 

Multilayer Convolutional Neural Network Architecture

Confidence Level Calculation

Confidence Calculation:

-	If prediction[0][0] >= 0.5: The result is "Fake".
-	If prediction[0][0] < 0.5: The result is "Real".
Confidence Calculation:
-	If the result is "Fake": confidence=prediction[0][0]\text{confidence} = \text{prediction}[0][0]confidence=prediction[0][0]
-	If the result is "Real": confidence=1−prediction[0][0]\text{confidence} = 1 - \text{prediction}[0][0]confidence=1−prediction[0][0]


