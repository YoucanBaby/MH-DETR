import torch
import cv2
import numpy as np

def visualize_attention(image, attention_weights, feature_map_size, alpha=0.5):
    # Compute the spatial attention map from the attention weights
    attention_map = attention_weights.sum(dim=0).view(feature_map_size, feature_map_size).detach().cpu().numpy()

    # Normalize the attention map to the range [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Resize the attention map to the input image size
    attention_map_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert the attention map to a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)

    # Overlay the heatmap on the input image
    visualized_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return visualized_image

# Load your image and run DETR model to get predictions and cross-attention weights
# ...

# Visualize the cross-attention for a specific object query
object_query_index = 0
feature_map_size = 32  # The spatial resolution of the feature map used in DETR
alpha = 0.5  # The weight for blending the heatmap and the input image


visualized_image = visualize_attention(image, cross_attention_weights[object_query_index], feature_map_size, alpha)

# Display the result
cv2.imshow("DETR Cross-Attention Visualization", visualized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
model.load_state_dict(checkpoint)