# MNIST Training with TensorFlow and Keras

## Overview
This project demonstrates how to train a deep learning model using TensorFlow and Keras on the MNIST dataset. The model is built using a simple fully connected neural network and achieves high accuracy on handwritten digit classification.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image has a size of 28x28 pixels and is grayscale.

## Dependencies
Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy
```

Alternatively, you can run this project in Google Colab using the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tj-mas04/mnist_training/blob/main/Model1)

## Model Architecture
The model consists of three fully connected layers:
- Input layer with 784 neurons (flattened 28x28 image)
- Hidden layer with 512 neurons (ReLU activation)
- Hidden layer with 256 neurons (ReLU activation)
- Output layer with 10 neurons (softmax activation for classification)

## Code Implementation
### 1. Importing Required Libraries
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
```

### 2. Loading and Preprocessing Data
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
```

### 3. Defining the Model
```python
model = keras.Sequential([
    keras.Input(shape=(28*28)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 4. Compiling the Model
```python
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
```

### 5. Training the Model
```python
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
```

### 6. Evaluating the Model
```python
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
```

### 7. Model Summary
```python
print(model.summary())
```

## Results
After training for 5 epochs, the model achieves an accuracy of around **98%** on the test dataset.

## Future Improvements
- Implementing Convolutional Neural Networks (CNNs) for better accuracy
- Using Data Augmentation techniques
- Experimenting with different optimizers and hyperparameters

## Author
**Sam T James**  
GitHub: [tj-mas04](https://github.com/tj-mas04)

## License
This project is open-source and available under the MIT License.

