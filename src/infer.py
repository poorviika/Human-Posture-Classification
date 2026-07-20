import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

CLASS_NAMES = ['bending', 'lying', 'sitting', 'standing']


def load_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert('RGB').resize(target_size)
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on a single posture silhouette image.'
    )
    parser.add_argument('--model', required=True, help='Path to the Keras model file')
    parser.add_argument('--image', required=True, help='Path to the input image')
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')
    if not image_path.exists():
        raise FileNotFoundError(f'Image file not found: {image_path}')

    model = keras.models.load_model(str(model_path))
    image = load_image(str(image_path))
    predictions = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    print(f'Predicted posture: {predicted_label}')
    print(f'Confidence: {confidence:.4f}')
    print('Raw probabilities:')
    for label, score in zip(CLASS_NAMES, predictions):
        print(f'  {label}: {score:.4f}')


if __name__ == '__main__':
    main()
