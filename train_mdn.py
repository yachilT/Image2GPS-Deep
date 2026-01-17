import torch
from Retrival_Dino_Salad.dinomdn import get_dino_mdn
from preprocess import GPSRectNorm, get_dataset



device = "cuda" if torch.cuda.is_available() else "cpu"

DINO_DIR_PATH = "Retrival_Dino_Salad"

def main():
    gps_norm, full_dataset, train_dataset, val_dataset, train_loader, val_loader = get_dataset()
    print("loading dino_mdn")
    dino_mlp = get_dino_mdn('trained_models/mdn_head2.pth', gps_norm, train_loader, val_loader)



if __name__ == '__main__':
    main()