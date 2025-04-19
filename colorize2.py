import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm

# Configuration
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 100

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(current_dir, 'data')

TRAIN_BLACK_PATH = os.path.join(DATA_ROOT, 'train_black')
TRAIN_COLOR_PATH = os.path.join(DATA_ROOT, 'train_color')
TEST_BLACK_PATH = os.path.join(DATA_ROOT, 'test_black')
TEST_COLOR_PATH = os.path.join(DATA_ROOT, 'test_color')

def verify_dataset_structure():
    print("\nVerifying dataset structure...")
    required_folders = {
        "Training Black": TRAIN_BLACK_PATH,
        "Training Color": TRAIN_COLOR_PATH,
        "Test Black": TEST_BLACK_PATH,
        "Test Color": TEST_COLOR_PATH
    }
    
    for name, path in required_folders.items():
        exists = os.path.exists(path)
        print(f"{name}: {path} - {'Exists' if exists else 'MISSING'}")
        if not exists:
            raise FileNotFoundError(f"Folder not found: {path}")
        
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Found {len(files)} images")
        if len(files) == 0:
            raise ValueError(f"No images found in {path}")

def load_image_pairs(black_dir, color_dir, max_samples=None):
    black_images = []
    color_images = []

    black_files = sorted(os.listdir(black_dir))
    color_files = sorted(os.listdir(color_dir))

    if max_samples:
        black_files = black_files[:max_samples]
        color_files = color_files[:max_samples]

    if max_samples is None:
        max_samples = min(len(black_files), len(color_files))

    for black_file, color_file in tqdm(zip(black_files, color_files), total=len(black_files)):
        black_path = os.path.join(black_dir, black_file)
        color_path = os.path.join(color_dir, color_file)

        grey_img = cv2.imread(black_path, cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(color_path)

        if grey_img is None or color_img is None:
            continue

        grey_img = cv2.resize(grey_img, IMAGE_SIZE)
        color_img = cv2.resize(color_img, IMAGE_SIZE)

        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = color_img.astype('float32') / 255.0
        lab_img = rgb2lab(color_img)

        grey_img = grey_img.astype('float32') / 255.0
        grey_img = np.expand_dims(grey_img, axis=-1)

        ab_channels = (lab_img[:, :, 1:] / 128).astype('float32')

        black_images.append(grey_img)
        color_images.append(ab_channels)

    return np.array(black_images), np.array(color_images)

def build_model():
    input_img = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)

    output = Conv2D(2, (3,3), activation='tanh', padding='same')(x)

    model = Model(input_img, output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return model

def colorize_image(model, black_image_path, original_color_path=None):
    grey_img = cv2.imread(black_image_path, cv2.IMREAD_GRAYSCALE)
    original_size = grey_img.shape[:2]
    grey_img = cv2.resize(grey_img, IMAGE_SIZE)
    grey_img = grey_img.astype('float32') / 255.0
    grey_img = np.expand_dims(grey_img, axis=-1)
    grey_img = np.expand_dims(grey_img, axis=0)

    ab_channels = model.predict(grey_img) * 128

    lab_img = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    lab_img[:,:,:,0] = grey_img[0,:,:,0] * 100
    lab_img[:,:,:,1:] = ab_channels[0]

    rgb_img = lab2rgb(lab_img[0])
    rgb_img = cv2.resize(rgb_img, (original_size[1], original_size[0]))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(black_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title('Input Black & White')
    plt.axis('off')

    if original_color_path and os.path.exists(original_color_path):
        plt.subplot(1, 3, 2)
        original_color = cv2.imread(original_color_path)
        original_color = cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB)
        plt.imshow(original_color)
        plt.title('Original Color')
        plt.axis('off')

    plt.subplot(1, 3, 3 if original_color_path else 2)
    plt.imshow(rgb_img)
    plt.title('Colorized')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    verify_dataset_structure()

    print("Loading training data...")
    X_train, Y_train = load_image_pairs(TRAIN_BLACK_PATH, TRAIN_COLOR_PATH)
    print(f"Loaded {len(X_train)} training pairs")

    print("Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    print("Training model...")
    history = model.fit(
        X_train, Y_train,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save('colorization_model.h5')
    print("Model saved as 'colorization_model.h5'")

    test_black_files = os.listdir(TEST_BLACK_PATH)[:3]
    test_color_files = os.listdir(TEST_COLOR_PATH)[:3]

    for black_file, color_file in zip(test_black_files, test_color_files):
        black_path = os.path.join(TEST_BLACK_PATH, black_file)
        color_path = os.path.join(TEST_COLOR_PATH, color_file)
        print(f"\nColorizing: {black_file}")
        colorize_image(model, black_path, color_path)

if __name__ == "__main__":
    main()
