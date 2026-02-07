#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


## 

"""
We're going to examine a few different models for classifying images in the CIFAR-10 dataset, 
starting with a 4-layer fully-connected model.
Each input image is 32x32 pixels with red/green/blue channels for a total size of 32x32x3 and 
each image corresponds to exactly one of 10 classes.
`flatten` layer
3 Dense layers of 128 units and leaky_relu activation, and
a final 10-unit Dense layer with no activation. 
 When compiling the model, be sure to use the "from_logits=True" flag, since the 
model output will be logits rather than class probabilities.

Create a function named build_model1() (above the main guard), which implements the above model 
in Keras.  Use your function to build the model and, name it 'model1'. Use the summary() method 
to check your parameter count and output size calculations.  (Make sure to 'compile' it).

"""
def build_model1():
  model = Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Flatten(),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(10)

  ])
  model.summary()
  return model
"""
Conv 2D: 32 filters, 3x3 kernel, stride=2 (in both x,y dimensions), "same" padding, "relu" activation.
BatchNorm
Conv 2D: 64 filters, 3x3 kernel, stride=2 (in both x,y dimensions),  "same" padding, "relu" activation.
BatchNorm
Four more pairs of Conv2D+Batchnorm, with no striding option (so stride defaults to 1). 
Conv 2D: 128 filters, 3x3 kernel, (in both x,y dimensions),  "same" padding
BatchNorm
Flatten
Dense (aka Fully Connected) , 10 units
"""
def build_model2():
  model = None # Add code to define model 1.
  model = Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10)
])
  model.summary()
  return model
"""
Model 2 but seperable 
"""
def build_model3():
  model = Sequential([
      layers.SeparableConv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10)
  ])
  model.summary()
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10)
  ])
  model.summary()
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
  train_images = train_images.astype('float32') / 255.0
  test_images = test_images.astype('float32') / 255.0

  # Split training into train/validation (80/20)
  split = int(0.8 * len(train_images))
  val_images, val_labels = train_images[split:], train_labels[split:]
  train_images, train_labels = train_images[:split], train_labels[:split]

  print(f"Training samples: {len(train_images)}")
  print(f"Validation samples: {len(val_images)}")
  print(f"Test samples: {len(test_images)}")
  ########################################
  ## Build and train model 1
  
  model1 = build_model1()
  model1.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
  history1 = model1.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=30,
        batch_size=64,
        verbose=1
    )
  test_loss1, test_acc1 = model1.evaluate(test_images, test_labels, verbose=0)
  print(f"Model 1 - Test accuracy: {test_acc1:.4f}")
  # Test Image
  test_img = np.array(keras.utils.load_img(
      './test-image.jpg',
      color_mode='rgb',
      target_size=(32,32))
  )/255.0
  test_img = np.expand_dims(test_img, axis=0)  # Add batch dimension
  pred = model1.predict(test_img)
  class_idx = np.argmax(pred, axis=1)[0]
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
  print(f"Predicted class: {class_names[class_idx]}")

  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  print("\n--- Model 2: CNN ---")
  model2 = build_model2()
  model2.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
 )
  history2 = model2.fit(
      train_images, train_labels,
      validation_data=(val_images, val_labels),
      epochs=30,
      batch_size=64,
      verbose=1
  )
  test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose=0)
  print(f"Model 2 - Test accuracy: {test_acc2:.4f}")
  
  ### Repeat for model 3 and your best sub-50k params model

  print("\n--- Model 3: CNN with Separable Convolutions ---")
  model3 = build_model3()
  model3.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
  history3 = model3.fit(
      train_images, train_labels,
      validation_data=(val_images, val_labels),
      epochs=30,
      batch_size=64,
      verbose=1
  )
  test_loss3, test_acc3 = model3.evaluate(test_images, test_labels, verbose=0)
  print(f"Model 3 - Test accuracy: {test_acc3:.4f}")

  print("\n--- Best Model (<50k params) ---")
  model50k = build_model50k()
  model50k.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
  history_best = model50k.fit(
      train_images, train_labels,
      validation_data=(val_images, val_labels),
      epochs=50,  # Train a bit longer
      batch_size=64,
      verbose=1
  )
  test_loss_best, test_acc_best = model50k.evaluate(test_images, test_labels, verbose=0)
  print(f"Best Model - Test accuracy: {test_acc_best:.4f}")
  if test_acc_best >= 0.60:
      print("✓ Achieved at least 60% accuracy.")
  else:
      print("✗ Did not reach 60% accuracy.")

  # Save the best model
  model50k.save("best_model.h5")
  print("Best model saved as 'best_model.h5'")

