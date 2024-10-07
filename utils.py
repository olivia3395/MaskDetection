import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

# Utility function to visualize an image

def imshow(img, title=None):
    """
    Display an image with an optional title.

    Args:
        img: Tensor representing the image.
        title: Title to display above the image (optional).
    """
    img = img / 2 + 0.5  # Unnormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Utility function to visualize feature maps

def visualize_feature_maps(model, image, layer_num):
    """
    Visualize the feature maps of a specific layer of the model for a given image.

    Args:
        model: PyTorch model.
        image: Input image tensor.
        layer_num: The layer number to visualize feature maps from.
    """
    model.eval()  # Set the model to evaluation mode
    x = image.unsqueeze(0).to(next(model.parameters()).device)  # Add batch dimension and match model device

    # Iterate through the model until the specified layer number
    for idx, layer in enumerate(model.children()):
        x = layer(x)
        if idx == layer_num:
            break

    # Get feature maps and visualize them
    feature_maps = x.squeeze(0).cpu().detach()  # Remove batch dimension and move to CPU
    fig, axs = plt.subplots(1, min(8, feature_maps.shape[0]), figsize=(20, 5))  # Display the first 8 feature maps
    for i in range(min(8, feature_maps.shape[0])):
        axs[i].imshow(feature_maps[i].numpy(), cmap='gray')
        axs[i].axis('off')
    plt.show()

# Utility function to visualize predictions

def visualize_predictions(model, loader, device, num_images=8):
    """
    Visualize a set of correct and incorrect predictions from the model.

    Args:
        model: PyTorch model.
        loader: DataLoader for the dataset.
        device: Device to run the model on.
        num_images: Number of images to display.
    """
    model.eval()
    images_shown = 0
    correct_images = []
    incorrect_images = []
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Collect correct and incorrect predictions
            for i in range(images.size(0)):
                if predicted[i] == labels[i]:
                    correct_images.append((images[i].cpu(), predicted[i].cpu(), labels[i].cpu()))
                else:
                    incorrect_images.append((images[i].cpu(), predicted[i].cpu(), labels[i].cpu()))

                images_shown += 1
                if images_shown >= num_images:
                    break
            if images_shown >= num_images:
                break

    # Visualize correctly classified images
    print("Correctly Classified Images:")
    for img, pred, true in correct_images[:4]:
        imshow(torchvision.utils.make_grid(img), title=f'Predicted: {pred.item()}, True: {true.item()}')

    # Visualize misclassified images
    print("Incorrectly Classified Images:")
    for img, pred, true in incorrect_images[:4]:
        imshow(torchvision.utils.make_grid(img), title=f'Predicted: {pred.item()}, True: {true.item()}')
