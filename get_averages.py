import argparse
from configs.cfg import get_cfg_defaults
from dataio.dataloader import get_dataloaders
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs/multi.yaml')
            
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.configs)

    train_loader, train_push_loader, val_loader, _, image_normalizer = get_dataloaders(cfg, print)


    # Iterate over the train_push_loader and calculate the mean and std of the images by channel
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    for i, ((_, img), _) in enumerate(train_push_loader):
        # Calculate the mean and std of the images by channel
        mean += img.mean(dim=(0,2,3)) * len(img)
        std += img.std(dim=(0,2,3)) * len(img)
        num_samples += len(img)

    mean /= num_samples
    std /= num_samples

    print(f"Mean: {mean}")
    print(f"Std: {std}")

if __name__ == '__main__':
    main()