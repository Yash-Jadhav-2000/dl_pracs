#########################################################################

Practical 1
This code trains a feed-forward neural network on the "Churn_Modelling.csv" dataset with two different optimizers: Adam and Adadelta. Here is a breakdown of the code:
The CSV dataset is read in using pandas, and the input features and target variable are extracted.
The third column of the input features is assumed to be a categorical variable, so it is encoded using LabelEncoder from scikit-learn.
The second column of the input features is assumed to be a categorical variable with multiple categories, so it is encoded using OneHotEncoder from scikit-learn. The remaining columns of the input features are standardized using StandardScaler from scikit-learn.
The standardized input features and target variable are split into training and test sets using train_test_split from scikit-learn.
A feed-forward neural network model with two hidden layers is defined using TensorFlow's Sequential API. The model has 6 units in each hidden layer and uses the ReLU activation function. The output layer has 1 unit and uses the sigmoid activation function.
The model is compiled using the Adam optimizer and binary crossentropy loss function, and accuracy is used as the evaluation metric. The model is trained on the training set for 100 epochs with a batch size of 32.
The model is recompiled using the Adadelta optimizer and trained on the same training set for another 100 epochs with a batch size of 32.
Overall, this code demonstrates how to train a feed-forward neural network using different optimizers and evaluate the performance of the model on a binary classification problem.



#########################################################################


Practical 2

This program implements regularization techniques Lasso and Ridge to prevent overfitting of the model.
First, the program imports necessary libraries such as numpy, matplotlib, pandas, and metrics from scikit-learn to evaluate the model's performance. It then reads in a CSV file named "salary_data.csv" using pandas and splits the input features (X) and target variable (y) into training and test sets using the train_test_split function from scikit-learn.
Next, the program creates a Lasso model and trains it on the training set using the fit method. It then calculates and prints the root mean squared error (RMSE) for both the training and test sets using metrics.mean_squared_error and np.sqrt.
The same process is repeated for the Ridge model, where a Ridge object is created and trained on the training set. The RMSE values for both the training and test sets are calculated and printed using metrics.mean_squared_error and np.sqrt.
Both Lasso and Ridge models are used for regularization by adding a penalty term to the loss function, which helps to reduce the model's coefficients' magnitude. Lasso uses L1 regularization, where the penalty term is the absolute sum of the coefficients, and Ridge uses L2 regularization, where the penalty term is the squared sum of the coefficients. By adding the penalty term, these regularization techniques help to prevent overfitting by reducing the model's complexity.
Overall, this program demonstrates how Lasso and Ridge regularization techniques can be used to prevent overfitting in a linear regression model.





#########################################################################


Practical 4
The MNIST dataset is loaded using Keras mnist.load_data() function.
The input data is normalized by dividing the pixel values by 255 to scale them between 0 and 1.
The autoencoder model is defined using the Keras functional API. It has one input layer of size 784 (the number of pixels in an image), one hidden layer of size 32 with ReLU activation function, and one output layer of size 784 with sigmoid activation function.
The autoencoder model is compiled with the binary cross-entropy loss function and the Adam optimizer.
The model is trained on the training set (XTrain) for 100 epochs with a batch size of 256.
The model's history is saved in history.
The trained model is used to reconstruct the test images (DecodedDigits).
Finally, the reconstructed images are displayed alongside their original counterparts using matplotlib.


#########################################################################


Practical 5

This code implements a convolutional neural network for digit recognition on the MNIST dataset. The MNIST dataset consists of handwritten digits from 0 to 9, and the goal is to correctly classify them.
The code first loads the dataset using the Keras library. It then preprocesses the data by normalizing the pixel values to be between 0 and 1 and converting the labels to one-hot encoded vectors.
The architecture of the CNN consists of two convolutional layers, followed by a max pooling layer, a dropout layer to prevent overfitting, a flattening layer to convert the output of the convolutional layers to a 1D vector, two fully connected layers, and an output layer with a softmax activation function to produce class probabilities.
The model is compiled using the Adadelta optimizer and the categorical cross-entropy loss function. The model is trained on the training set for 12 epochs with a batch size of 500.
After training, the model is evaluated on the test set, and the loss and accuracy are printed. Finally, the model is used to predict the class labels for the test set.



#########################################################################


Practical 6

The code imports the necessary packages and libraries, including TensorFlow and Keras.
It sets up an ImageDataGenerator to preprocess the data for training and testing the CNN model.
The dataset directory is specified, and the ImageDataGenerator is applied to the training and testing sets to generate batches of images and labels.
A Sequential model is defined and created using Keras.
The model consists of multiple layers including two convolutional layers, two max pooling layers, a flatten layer, and two dense layers.
The model is compiled with the Adam optimizer and binary cross-entropy loss function, and accuracy is used as a metric to evaluate performance during training.
The model is trained on the training set and validated on the testing set for 25 epochs.
A single image is loaded for prediction using the model, and the result is outputted as either "dog" or "cat" depending on the predicted label.
Overall, this code demonstrates how to use transfer learning to classify images in a dataset using a pre-trained CNN model. The model is trained on the dataset, and then tested on a new image to make a prediction.


#########################################################################


Practical 7

This code implements three different forms of recurrent neural networks (RNNs): SimpleRNN, GRU, and LSTM. It uses the Keras library, which is a high-level neural network API that runs on top of TensorFlow.
The dataset used in this code is the IMDB movie review dataset, which consists of 50,000 movie reviews that are classified as either positive or negative. The dataset is preprocessed so that each review is represented as a sequence of integers, where each integer represents a word in the review. The integer sequences are padded so that they all have the same length.
The RNN models are trained on this dataset using binary cross-entropy loss and the Adam optimizer. Each model is trained for 5 epochs with a batch size of 64.
After training, the models are evaluated on the test set using the evaluate() method. The score reported is the binary cross-entropy loss and the accuracy metric.
Overall, this code provides a good starting point for learning how to implement RNNs in Keras. However, it is important to note that the performance of the models can be improved by experimenting with different hyperparameters and architectures.


#########################################################################


Practical 8

This is a Python program for object detection from an image using the Single Shot MultiBox Detector (SSD) algorithm. It uses the OpenCV library to perform the detection.
The program starts by importing the required libraries: OpenCV and matplotlib. It then specifies the location of the configuration file and the frozen model file. The DetectionModel() function is used to load the frozen model and its configuration.
Next, it reads the class labels from a file named 'labels.txt'. The rstrip() function is used to remove any newline characters at the end of each line, and the split() function is used to split the lines into a list of strings. The list of class labels is then printed to the console, along with its length.
The setInputSize() function is used to set the input size of the image to (320, 320). The setInputScale() function is used to scale the pixel values of the image by 1/127.5, and the setInputMean() function is used to subtract (127.5, 127.5, 127.5) from the pixel values. Finally, setInputSwapRB() function is used to swap the color channels of the image from BGR to RGB.
The program then loads an image named "dog.jpg" using OpenCV's imread() function and converts it from BGR to RGB format using cvtColor() function. The original image is displayed using matplotlib's imshow() function.
The detect() function is then used to perform object detection on the image. The confThreshold parameter is set to 0.5 to filter out low-confidence detections. The detect() function returns three values: ClassIndex, Confidence, and bbox. ClassIndex is an array containing the class IDs of the detected objects, Confidence is an array containing the detection confidence scores, and bbox is an array containing the bounding box coordinates of the detected objects.
The program then sets the font scale and font type for the text to be displayed on the image. It then loops through each detected object and draws a rectangle around it using cv2.rectangle() function. The class label for the object is displayed next to the rectangle using the cv2.putText() function. The final image with the bounding boxes and class labels is displayed using matplotlib's imshow() function.


