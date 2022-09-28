#!/usr/bin/env python
# coding: utf-8
#----------------------------------------------------------------------------
# @gilbellosta, 2022-08-19
# Style transfer - using "full" MMD distances between distributions
#----------------------------------------------------------------------------

from tensorflow import keras
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

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

total_variation_weight = 0.01
style_weight = 10000.1
content_weight = 1.1

# Loss functions (using MMD)


def content_loss(x, y):
    return tf.reduce_mean(tf.square(x - y))

def mean_kernel_product(x1, x2, o1, o2, sigma = 10):

    tmp1 = tf.gather(x1, o1)
    tmp2 = tf.gather(x2, o2)

    tmp = tmp1 - tmp2
    tmp = tf.reduce_sum(tf.square(tmp), 1)

    sigma = tf.math.reduce_std(tmp)

    tmp = tf.exp(-tmp / (2 * sigma * sigma))
    return tf.reduce_mean(tmp)

def style_loss(x1, x2, o1, o2):
    x1_s = tf.reshape(tf.sigmoid(x1), [-1, x1.shape[2]])
    x2_s = tf.reshape(tf.sigmoid(x2), [-1, x2.shape[2]])

    loss =  - 2 * mean_kernel_product(x1_s, x2_s, o1, o2)
    loss += mean_kernel_product(x2_s, x2_s, o1, o2)
    loss += mean_kernel_product(x1_s, x1_s, o1, o2)

    return loss

def total_variation_loss(x):
    a = tf.square(x[:, : img_height - 1, : img_width - 1, :] - x[:, 1:, : img_width - 1 , :])
    b = tf.square(x[:, : img_height - 1, : img_width - 1, :] - x[:, : img_height - 1, 1:, :])
    return tf.reduce_mean(tf.pow(a + b, 1.25))


# Preprocessing the base image

base_image = preprocess_image(base_image_path)
base_image_features = feature_extractor(base_image)[content_layer_name][0, :, :, :]

# Preprocessing the style image

style_reference_image = preprocess_image(style_reference_image_path)
style_reference_features = feature_extractor(style_reference_image)
style_reference_features = {
    layer_id : style_reference_features[layer_id][0,:,:,]
    for layer_id in style_layer_names
}

reshuffle01 = {k : np.random.choice(v.shape[0] * v.shape[1], 100000, replace = True) for k, v in style_reference_features.items() }
reshuffle02 = {k : np.random.choice(v.shape[0] * v.shape[1], 100000, replace = True) for k, v in style_reference_features.items() }


def compute_loss(combination_image, base_image_features, style_reference_features):

    loss = tf.zeros(shape=())
    features = feature_extractor(combination_image)

    # loss associated to the image shape
    loss = loss + content_weight * content_loss(
        base_image_features, features[content_layer_name][0, :, :, :]
    )

    print(f"shape: {loss}")

    for layer_name in style_layer_names:
        layer_style_reference_features = style_reference_features[layer_name]
        combination_features = features[layer_name][0, :, :, :]

        style_loss_value = style_loss(
            combination_features,
            layer_style_reference_features,
            reshuffle01[layer_name],
            reshuffle02[layer_name]
        )
        loss += (style_weight / len(style_layer_names)) * style_loss_value

    print(f"style: {loss}")

    return loss


# In[ ]:


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
        fname = f"combination_image_at_iteration_{i}_mmd.png"
        keras.utils.save_img(fname, img)

