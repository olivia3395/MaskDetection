import matplotlib.pyplot as plt

def visualize_filters(layer):
    """
    Visualize the convolutional filters of a given convolutional layer.
    
    Args:
        layer: A convolutional layer from a PyTorch model.
    """
    filters = layer.weight.data.cpu().numpy()  # Get the weights of the convolutional filters
    fig, axs = plt.subplots(1, min(8, filters.shape[0]), figsize=(20, 5))  # Display the first 8 filters
    for i in range(min(8, filters.shape[0])):
        axs[i].imshow(filters[i, 0], cmap='gray')  # Visualize only the first channel of each filter
        axs[i].axis('off')
    plt.show()

# Example usage:
# conv_layer = model.conv1
# visualize_filters(conv_layer)
