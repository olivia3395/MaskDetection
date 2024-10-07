from torch.utils.data import DataLoader
from model.mask_dataset import MaskDataset
from torchvision import transforms

def get_dataloaders(train_csv, train_root, valid_csv, valid_root, test_csv, test_root, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = MaskDataset(csv_file=train_csv, root_dir=train_root, transform=transform)
    valid_dataset = MaskDataset(csv_file=valid_csv, root_dir=valid_root, transform=transform)
    test_dataset = MaskDataset(csv_file=test_csv, root_dir=test_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader
