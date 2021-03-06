import pandas as pd
from matplotlib import cm
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.express as px

tf.config.threading.set_inter_op_parallelism_threads(1)

model_builder = keras.applications.efficientnet.EfficientNetB0
img_size = (224, 224)
preprocess_input = keras.applications.efficientnet.preprocess_input
decode_predictions = keras.applications.efficientnet.decode_predictions


model = model_builder(weights="imagenet")

layer_names = ["stem_activation",
               "block1a_project_bn",
               "block2a_project_bn",
               "block2b_add",
               "block3a_project_bn",
               "block3b_add",
               "block4a_project_bn",
               "block4b_add",
               "block4c_add",
               "block5a_project_bn",
               "block5b_add",
               "block5c_add",
               "block6a_project_bn",
               "block6b_add",
               "block6c_add",
               "block6d_add",
               "block7a_project_bn",
               "top_activation"]


def get_img_array(img):
    img = img.convert('RGB')

    img = img.resize(img_size)

    array = keras.preprocessing.image.img_to_array(img)

    array = np.expand_dims(array, axis=0)
    return array


def extract_predictions(img_array):
    img_array = get_img_array(img_array)

    preds = model.predict(img_array)

    # get 5 highest classes
    resulting_classes = decode_predictions(preds, top=5)
    df = pd.DataFrame(columns=["class", "confidence"])
    i = 0
    for c in resulting_classes[0]:
        df.loc[i] = [c[1].replace("_", " ").title(), round(
            float(c[2]), 3)]  # round for better visibility
        i += 1
    return df


def make_gradcam_heatmap(img_array, pred_index=None, layer_index=-1):
    layer_name = layer_names[layer_index]
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        else:
            selected_pred = tf.squeeze(preds)
            top_five_classes = tf.argsort(selected_pred)[-5:][::-1]
            pred_index = top_five_classes[pred_index]
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def make_gradcam_output(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    img_array = keras.preprocessing.image.img_to_array(img)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    img_array = img_array[:, :, :3]
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


def gradcam(img, selected_class=None, layer_index=-1):
    img = img.convert('RGB')
    img_array = get_img_array(img)
    img_array_pre = preprocess_input(img_array.copy())
    heatmap = make_gradcam_heatmap(img_array_pre, selected_class, layer_index)
    img = make_gradcam_output(img, heatmap)
    return img
