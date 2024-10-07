import matplotlib.pyplot as plt

def visualize_feature_maps(model, image, layer_num):
    model.eval()
    x = image.unsqueeze(0).to(device)

    for idx, layer in enumerate(model.children()):
        x = layer(x)
        if idx == layer_num:
            break

    feature_maps = x.squeeze(0).cpu().detach()
    fig, axs = plt.subplots(1, 8, figsize=(20, 5))
    for i in range(8):
        axs[i].imshow(feature_maps[i].numpy(), cmap='gray')
        axs[i].axis('off')
    plt.show()
