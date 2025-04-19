import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# Configuration
IMAGE_SIZE = (128, 128)  # Smaller size for faster training
BATCH_SIZE = 16
EPOCHS = 50  # You can increase this for better results
DATASET_PATH = "small_dataset"  # Create this folder and add some color images

# Create dataset folder if it doesn't exist
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
    print(f"Please add some color images to the {DATASET_PATH} folder and run again.")
    exit()

# Load and preprocess images
def load_images():
    images = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(DATASET_PATH, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR by default
            img = cv2.resize(img, IMAGE_SIZE)
            img = img.astype('float32') / 255.0  # Normalize to [0,1]
            images.append(img)
    
    if len(images) == 0:
        print(f"No images found in {DATASET_PATH}. Please add some color images.")
        exit()
    
    return np.array(images)

# Prepare training data
def prepare_data(images):
    # Convert RGB to grayscale (this will be our input)
    X = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    
    # The original color images will be our target
    Y = images
    
    return X, Y

# Build the model
def build_model():
    input_img = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # 3 channels for RGB
    
    model = Model(input_img, output_img)
    model.compile(optimizer=Adam(), loss='mse')
    
    return model

# Train the model
def train_model(model, X, Y):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X, Y,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )
    
    return history

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss During Training')
    
    plt.show()

# Colorize a new image
def colorize_image(model, image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img = cv2.resize(img, IMAGE_SIZE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Normalize and reshape for model
    img_gray = img_gray.astype('float32') / 255.0
    img_gray = np.expand_dims(img_gray, axis=-1)
    img_gray = np.expand_dims(img_gray, axis=0)
    
    # Predict color
    colorized = model.predict(img_gray)[0]
    
    # Resize back to original dimensions
    colorized = cv2.resize(colorized, (original_size[1], original_size[0]))
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(colorized)
    plt.title('Colorized')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Save the model after training
def save_model(model, filename='colorization_model.h5'):
    model.save(filename)
    print(f"Model saved to {filename}")

# Main function
def main():
    # Load and prepare data
    print("Loading images...")
    images = load_images()
    X, Y = prepare_data(images)
    print(f"Loaded {len(images)} images of size {IMAGE_SIZE}")
    
    # Build and train model
    print("Building model...")
    model = build_model()
    model.summary()
    
    print("Training model...")
    history = train_model(model, X, Y)
    
    # Plot training progress
    plot_history(history)
    
    # Save the trained model
    save_model(model)
    
    # Test colorization
    while True:
        test_image = input("Enter path to an image to colorize (or 'quit' to exit): ")
        if test_image.lower() == 'quit':
            break
        
        if os.path.exists(test_image):
            colorize_image(model, test_image)
        else:
            print("File not found. Please try again.")

if __name__ == "__main__":
    main()
