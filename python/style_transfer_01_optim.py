#!/usr/bin/env python
# coding: utf-8
#----------------------------------------------------------------------------
# @gilbellosta, 2022-08-19
# Style transfer - basic implementation - using standard Gram matrix
#----------------------------------------------------------------------------

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

url_base  = 'https://img-datasets.s3.amazonaws.com/sf.jpg'
url_style = 'https://img-datasets.s3.amazonaws.com/starry_night.jpg'

base_image_path = keras.utils.get_file('sf.jpg', origin=url_base)
style_reference_image_path = keras.utils.get_file('starry_night.jpg', origin=url_style)

original_width, original_height = keras.utils.load_img(base_image_path).size
img_height = 400
img_width = round(original_width * img_height / original_height)


def preprocess_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_image(img):
    img = img.reshape((img_height, img_width, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


# We will use VGG19 as the underlying NN.

model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

content_layer_name = "block5_conv2"

# Weights to calibrate style transfer level

total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8


# Loss functions (as per the original paper)

def content_loss(x, y):
    return tf.reduce_sum(tf.square(x - y))

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style_img, combination_img):
    S = style_img
    C = gram_matrix(combination_img)
    channels = 3
    size = img_height * img_width
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = tf.square(x[:, : img_height - 1, : img_width - 1, :] - x[:, 1:, : img_width - 1 , :])
    b = tf.square(x[:, : img_height - 1, : img_width - 1, :] - x[:, : img_height - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


# Preprocessing the base image

base_image = preprocess_image(base_image_path)
base_image_features = feature_extractor(base_image)[content_layer_name][0, :, :, :]

# Preprocessing the style image
# Note that the Gram matrix is precalculated

style_reference_image = preprocess_image(style_reference_image_path)
style_reference_features = feature_extractor(style_reference_image)
style_reference_features = {
    layer_id : gram_matrix(style_reference_features[layer_id][0,:,:,])
    for layer_id in style_layer_names
}


def compute_loss(combination_image, base_image_features, style_reference_features):

    loss = tf.zeros(shape=())
    features = feature_extractor(combination_image)

    # loss associated to the image shape
    loss = loss + content_weight * content_loss(
        base_image_features, features[content_layer_name][0, :, :, :]
    )

    for layer_name in style_layer_names:
        layer_style_reference_features = style_reference_features[layer_name]
        combination_features = features[layer_name][0, :, :, :]

        style_loss_value = style_loss(layer_style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * style_loss_value

    loss += total_variation_weight * total_variation_loss(combination_image)

    return loss

@tf.function
def compute_loss_and_grads(combination_image, base_image_features, style_reference_features):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image_features, style_reference_features)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 100.0, decay_steps = 100, decay_rate = 0.96
    )
)

combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image_features, style_reference_features
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss:.2f}")
        img = deprocess_image(combination_image.numpy())
        fname = f"combination_image_at_iteration_{i}_optimized_code.png"
        keras.utils.save_img(fname, img)

