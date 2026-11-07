
import gradio as gr
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
model = keras.models.load_model(
    "mobilenetv2_finetuned.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

# Grad-CAM function
def make_gradcam_heatmap(img_array):
    base_model = model.get_layer("mobilenetv2_1.00_224")

    conv_model = keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.get_layer("out_relu").output
    )

    rescaling = model.get_layer("rescaling")
    preprocessing = model.get_layer("lambda")
    pooling = model.get_layer("global_average_pooling2d")
    dense = model.get_layer("dense_2")
    final_layer = model.get_layer("dense_3")

    with tf.GradientTape() as tape:
        x = rescaling(img_array)
        x = preprocessing(x)
        conv_outputs = conv_model(x, training=False)
        tape.watch(conv_outputs)

        x = pooling(conv_outputs)
        x = dense(x)
        predictions = final_layer(x)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads, axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# Prediction function
def predict_image(image):

    if image is None:
        return {"No image uploaded": 1.0}, None

    image = image.convert("RGB")
    resized = image.resize((224, 224))

    img_array = np.array(resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "Not Laptop"
        confidence = float(prediction)
    else:
        label = "Laptop"
        confidence = float(1 - prediction)

    result = {label: confidence}

    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    heatmap_colored = cv2.cvtColor(
        heatmap_colored,
        cv2.COLOR_BGR2RGB
    )

    original = img_array[0]

    overlay = 0.6 * original + 0.4 * heatmap_colored / 255.0
    overlay = np.clip(overlay, 0, 1)

    overlay = (overlay * 255).astype("uint8")
    overlay_image = Image.fromarray(overlay)

    return result, overlay_image


# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil",
        label="Upload an Image"
    ),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Image(label="Grad-CAM Explanation")
    ],
    title="Laptop vs Not Laptop Classifier",
    description="Upload an image to classify it and visualize which image regions influenced the prediction."
)

# Launch application
demo.launch()
