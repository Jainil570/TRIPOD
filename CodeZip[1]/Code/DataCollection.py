import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the CSV file containing the image file names and labels
csv_file_path = r"C:\Users\hp\Desktop\dataset\csv.csv"  # Update with your path
data_info = pd.read_csv(csv_file_path)

# Load and preprocess images
def load_image(file_path):
    # Read the image
    img = cv2.imread(file_path)
    if img is None:
        print(f"Warning: Unable to load image at {file_path}. Skipping.")
        return None  # Return None if the image cannot be loaded
    # Resize the image to 224x224
    img_resized = cv2.resize(img, (224, 224))
    return img_resized

# Prepare data and labels
data = []
labels = []
for index, row in data_info.iterrows():
    file_path = os.path.join(r"C:\Users\hp\Desktop\dataset\images", row['file_name'])  # Update with your path
    image = load_image(file_path)
    if image is not None:  # Only add images that were successfully loaded
        data.append(image)
        labels.append(row['labels'])

# Check if any images were loaded
if len(data) == 0:
    raise ValueError("No images were loaded. Please check your image paths.")

data = np.array(data)  # Convert to numpy array
labels = pd.factorize(labels)[0]  # Convert labels to numerical values

# Reshape the data for CNN
data = data.reshape(len(data), 224, 224, 3)  # Reshape the data into image format

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(labels)), activation='softmax')  # Output layer
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the CNN model
test_loss, test_acc = cnn_model.evaluate(x_test, y_test)
print(f"CNN Model Accuracy: {test_acc * 100:.2f}%")

# Save CNN model
cnn_model.save(r'C:\Users\hp\Desktop\cnn_gesture_model.h5')
