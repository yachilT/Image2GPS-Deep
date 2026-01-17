import torch
from Retrival_Dino_Salad.dinomlp import get_dino_mlp
from preprocess import get_dataset



device = "cuda" if torch.cuda.is_available() else "cpu"

DINO_DIR_PATH = "Retrival_Dino_Salad"

def main():
    gps_norm, full_dataset, train_dataset, val_dataset, train_loader, val_loader = get_dataset()
    dino_mlp = get_dino_mlp('trained_models/mlp_head3.pth', train_loader, val_loader)



if __name__ == '__main__':
    main()