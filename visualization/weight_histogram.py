import matplotlib.pyplot as plt

def plot_weight_histograms(model):
    """
    Plot the weight distribution histograms for each layer of a given model.
    
    Args:
        model: A PyTorch model whose weights are to be visualized.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.data.size()) > 1:  # Only plot weights with more than 1 dimension
            plt.hist(param.data.cpu().numpy().flatten(), bins=30)
            plt.title(f'Weight distribution for layer: {name}')
            plt.xlabel('Weight')
            plt.ylabel('Frequency')
            plt.show()

# Example usage:
# plot_weight_histograms(model)
