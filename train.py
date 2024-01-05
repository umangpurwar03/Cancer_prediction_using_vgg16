import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import torch

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

# Function to augment images in each class
def augment_images(folder_path, target_count):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    classes = os.listdir(folder_path)

    for class_name in classes:
        class_folder = os.path.join(folder_path, class_name)
        class_images = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
        
        current_count = len(class_images)
        augmentation_factor = max(0, target_count - current_count)
        
        if augmentation_factor > 0:
            for i in range(augmentation_factor):
                sample_image_path = random.choice(class_images)
                sample_image = Image.open(sample_image_path)
                sample_image = sample_image.resize((150, 150))
                sample_image = np.array(sample_image)
                
                if len(sample_image.shape) == 2:
                    sample_image = np.expand_dims(sample_image, axis=2)
                    sample_image = np.repeat(sample_image, 3, axis=2)
                
                augmented_img = train_datagen.random_transform(sample_image)
                augmented_img = Image.fromarray(augmented_img.astype('uint8'))
                augmented_img.save(os.path.join(class_folder, f"augmented_{current_count + i}.png"))

# Function to create train and test datasets
def create_train_test_datasets(folder_path, test_size=0.2):
    images = []
    labels = []
    classes = os.listdir(folder_path)

    for class_name in classes:
        class_folder = os.path.join(folder_path, class_name)
        class_images = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
        images.extend(class_images)
        labels.extend([class_name] * len(class_images))

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    train_df = pd.DataFrame({'filename': train_images, 'class': train_labels})
    test_df = pd.DataFrame({'filename': test_images, 'class': test_labels})

    return train_df, test_df

# Function to define and compile the model
def build_compile_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Function to train the model
def train_model(model, train_generator, test_generator, epochs):
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=test_generator,
                        callbacks=[checkpoint])
    
    return history

# Function to plot training history
def plot_history(history):
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

# Main code
folder_path = 'ultrasound_data copy'
target_count = 5000
input_shape = (150, 150, 3)

# Display images from each class
classes = os.listdir(folder_path)
for class_name in classes:
    display_images_from_class(folder_path, class_name)

# Augment images
augment_images(folder_path, target_count)

# Create train and test datasets
train_df, test_df = create_train_test_datasets(folder_path)

# Create data generators
batch_size = 1
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'
)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'
)

# Build and compile the model
num_classes = len(classes)
model = build_compile_model(input_shape, num_classes)

# Train the model
epochs = 10
history = train_model(model, train_generator, test_generator, epochs)

# Plot training history
plot_history(history)

# Evaluate the model on test data
_, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


