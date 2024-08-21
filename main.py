import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2

dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []

    for image in root.findall('image'):
        image_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))

        mask = np.zeros((height, width), dtype=np.uint8)
        
        for polygon in image.findall('polygon'):
            points = polygon.get('points')
            points = np.array([[float(coord) for coord in point.split(',')] for point in points.split(';')], dtype=np.int32)
            cv2.fillPoly(mask, [points], color=1)  # Label "moor" diisi dengan 1

        annotations.append((image_name, mask))
    
    return annotations

def load_custom_dataset(data_dir, annotations_file):
    annotations = parse_annotations(annotations_file)
    images = []
    masks = []

    for image_name, mask in annotations:
        image_path = os.path.join(data_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

        image, mask = normalize(image, mask)

        images.append(image)
        masks.append(mask)

    return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(masks)))

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_and_augment(datapoint):
    input_image, input_mask = load_image(datapoint)
    input_image = tf.image.random_flip_left_right(input_image)
    input_mask = tf.image.random_flip_left_right(input_mask)
    return input_image, input_mask

annotation_file = 'data/annotations.xml'
image_dir = 'data/imgs'
custom_dataset = load_custom_dataset(image_dir, annotation_file)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 128
BUFFER_SIZE = 1000
STEP_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_images = dataset['train'].map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_layers = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_layers(labels)
        return inputs, labels
    
train_batches = (
    custom_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_images.batch(BATCH_SIZE)

def display(display_list):
    plt.figure(figsize=(15, 7))
    title = ["Gambar", "Tegalan", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        img = display_list[i]
        if len(img.shape) == 2:
            img = tf.expand_dims(img, axis=-1)
            
        if i == 1:
            img = np.fliplr(img)
            img = np.flipud(img)
        
        img = tf.clip_by_value(img, 0, 1)
        plt.imshow(tf.keras.utils.array_to_img(img))
        plt.axis('off')
    plt.show()
    
for images, masks in train_batches.take(5):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])
  
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)