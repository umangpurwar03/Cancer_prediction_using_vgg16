import torch
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
print('load')

# Check if GPU is available and set it up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:",device)

# Function to display images from each class
def display_images_from_class(folder_path, class_name, num_images=5):
    class_folder = os.path.join(folder_path, class_name)
    images = os.listdir(class_folder)
    selected_images = random.sample(images, num_images)
    
    plt.figure(figsize=(15, 3))
    for i, image_name in enumerate(selected_images):
        image_path = os.path.join(class_folder, image_name)
        img = Image.open(image_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.show()

# Display 5 images from each class
folder_path = r'ultrasound_data copy 2'
classes = os.listdir(folder_path)
for class_name in classes:
    display_images_from_class(folder_path, class_name)

# Create lists for images and labels
images = []
labels = []
for class_name in classes:
    class_folder = os.path.join(folder_path, class_name)
    class_images = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
    images.extend(class_images)
    labels.extend([class_name] * len(class_images))

# Split dataset into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create dataframes from lists of train and test images and labels
train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})
test_df = pd.DataFrame({'filename': test_images, 'class': test_labels})

# Augment the training dataset using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the pre-trained VGG16 model (without the top classification layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the pre-trained model
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(classes), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create data generators using flow_from_dataframe
batch_size = 32
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'
)

test_datagen = ImageDataGenerator()  # No augmentation for test data
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'
)

# Define a callback to save the best model during training
checkpoint = ModelCheckpoint('best_model_1.h5',  # Filepath to save the model
                             monitor='val_accuracy',  # Monitor validation accuracy
                             save_best_only=True,  # Save only the best model
                             mode='max',  # Mode to maximize validation accuracy
                             verbose=1)  # Verbosity mode (optional)

# Train the model with the ModelCheckpoint callback
epochs = 50
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    callbacks=[checkpoint])  # Include the checkpoint callback

# Plot training history (accuracy and loss)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the best model on test data (optional)
best_model = tf.keras.models.load_model('best_model_1.h5')
_, test_accuracy = best_model.evaluate(test_generator)
print(f"Test Accuracy (Best Model): {test_accuracy * 100:.2f}%")
