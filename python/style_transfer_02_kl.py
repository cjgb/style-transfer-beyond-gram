#!/usr/bin/env python
# coding: utf-8
#----------------------------------------------------------------------------
# @gilbellosta, 2022-08-19
# Style transfer - using KL-distance for style
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

total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8


# Loss functions using kl distance

def content_loss(x, y):
    return tf.reduce_sum(tf.square(x - y))

def style_loss(x1, x2):

    tmp  = x1['mu'] - x2['mu']
    tmp1 = tf.linalg.matvec(x2['sigma_inverse'], tmp)
    tmp  = tf.reduce_sum(tmp * tmp1)

    tmp1 = tf.linalg.matmul(x2['sigma_inverse'], x1['sigma'])
    tmp = tmp + tf.linalg.trace(tmp1)

    tmp = tmp - x1['sigma_log_det']
    tmp = tmp + x2['sigma_log_det']
    tmp = tmp - x1['mu'].shape[0]

    return tmp

def total_variation_loss(x):
    a = tf.square(x[:, : img_height - 1, : img_width - 1, :] - x[:, 1:, : img_width - 1 , :])
    b = tf.square(x[:, : img_height - 1, : img_width - 1, :] - x[:, : img_height - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))


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


# De cada matriz necesito:
# * La media
# * La matriz de covarianzas
# * Su inversa
# * Su determinante (log)


def get_mu(x):
    return tf.reduce_sum(x, (0,1))

def get_sigma(x):
    tmp = tf.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    tmp = tfp.stats.covariance(tmp)
    #TODO: Multiply I by a factor?
    return tmp + tf.eye(tmp.shape[0])

def get_inverse(x):
    return tf.linalg.inv(x)

def get_log_determinant(x):
    return tf.reduce_sum(tf.math.log(tf.math.real(tf.linalg.eigvals(x))))

def get_summary(x, add_inverse = True):
    out = {}
    out['mu'] = get_mu(x)
    out['sigma'] = get_sigma(x)
    out['sigma_log_det'] = get_log_determinant(out['sigma'])

    if add_inverse:
        out['sigma_inverse'] = get_inverse(out['sigma'])

    return out

style_reference_features = {k : get_summary(v) for k,v in style_reference_features.items()}


def compute_loss(combination_image, base_image_features, style_reference_features):

    loss = tf.zeros(shape=())
    features = feature_extractor(combination_image)

    # loss associated to the image shape
    loss = loss + content_weight * content_loss(
        base_image_features, features[content_layer_name][0, :, :, :]
    )

    for layer_name in style_layer_names:
        layer_style_reference_features = style_reference_features[layer_name]
        combination_features = get_summary(features[layer_name][0, :, :, :], add_inverse=False)

        style_loss_value = style_loss(combination_features, layer_style_reference_features)
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
        fname = f"combination_image_at_iteration_{i}_kl.png"
        keras.utils.save_img(fname, img)

