import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """
    Computes the Grad-CAM heatmap by tracing through nested models manually.
    """
    # 1. Find the base model (MobileNetV2) inside our model
    base_model = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower() or isinstance(layer, tf.keras.Model):
            base_model = layer
            break
            
    if base_model is None:
        base_model = model

    # 2. Find the last conv layer if not specified
    if last_conv_layer_name is None:
        # For MobileNetV2, 'Conv_1' is the last conv layer before global pooling
        last_conv_layer_name = 'Conv_1'

    # 3. Use GradientTape to trace through the nested layers
    # We must trace FROM the base model output TO the final prediction
    # AND FROM the base model input TO the conv output
    with tf.GradientTape() as tape:
        # Get conv layer output from base model
        # We need a model that maps base_model input -> [conv_output, base_model_output]
        inner_model = tf.keras.models.Model(
            inputs=[base_model.input],
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
        
        # Apply the inner model to the input
        conv_output, base_out = inner_model(img_array)
        
        # Now apply the rest of the head to base_out manually
        # This mirrors model.py head
        x = model.get_layer('gap')(base_out)
        x = model.get_layer('fc1')(x)
        x = model.get_layer('dropout')(x, training=False)
        preds = model.get_layer('output')(x)
        
        # Get target class gradients
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Compute gradients with respect to conv output
    grads = tape.gradient(top_class_channel, conv_output)
    
    # Standard Grad-CAM pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image.
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img

def interpret_results(heatmap):
    """
    Helper to provide textual interpretation of where the model focused.
    """
    print("\n[INFO] Interpreting Grad-CAM:")
    print("- High intensity areas (Red) show what the model 'looked' at.")
    print("- Correct focus: Hollow cheeks, sunken eyes, thin arms, narrow shoulders.")
    print("- REJECT model if heatmap highlights background or distractors.")
