import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for each class of images
class_dirs = ["BacterialBlight", "LeafSmut", "BrownSpot", "Healthy"]

# Data directory where the images are stored
image_dir = "rice_leaf_diseases"

# Create an ImageDataGenerator instance for training and validation with more robust data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=30,       # Randomly rotate images by up to 30 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,         # Apply random shear transformations
    zoom_range=0.2,          # Apply random zoom
    horizontal_flip=True,    # Randomly flip images horizontally
    vertical_flip=True,      # Randomly flip images vertically
    validation_split=0.2     # Reserve 20% of the data for validation
)

# Load training data using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=(256, 256),  # Resize images to the required input size
    batch_size=32,
    class_mode='categorical',
    subset='training'        # Set as training data
)

# Load validation data using ImageDataGenerator
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = validation_datagen.flow_from_directory(
    image_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'      # Set as validation data
)

# Build an enhanced CNN model with batch normalization and dropout for regularization
def build_enhanced_cnn_model(input_shape=(256, 256, 3), num_classes=4):
    model = models.Sequential()

    # 1st Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())  # Batch normalization after the Conv layer
    model.add(layers.MaxPooling2D((2, 2)))

    # 2nd Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # 3rd Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # 4th Convolutional Block
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output and add Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting

    # Output layer for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Build the enhanced CNN model
cnn_model = build_enhanced_cnn_model(input_shape=(256, 256, 3), num_classes=len(class_dirs))

# Print model summary
cnn_model.summary()

# Define callbacks for learning rate adjustment and early stopping
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
learning_rate_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model using the generators
history = cnn_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,  # You can adjust the number of epochs based on your needs
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=[early_stopping, learning_rate_scheduler]  # Add callbacks
)

# Save the model
cnn_model.save("rice_leaf_disease_classifier_optimized.h5")

# Evaluate the model on the validation data
val_loss, val_acc = cnn_model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
