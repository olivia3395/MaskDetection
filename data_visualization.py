import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img, title=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()

def visualize_predictions(model, loader, device, num_images=8):
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

    print("Correctly Classified Images:")
    for img, pred, true in correct_images[:4]:
        imshow(torchvision.utils.make_grid(img), title=f'Predicted: {pred.item()}, True: {true.item()}')

    print("Incorrectly Classified Images:")
    for img, pred, true in incorrect_images[:4]:
        imshow(torchvision.utils.make_grid(img), title=f'Predicted: {pred.item()}, True: {true.item()}')
