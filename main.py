import torch
from torch.utils.data import DataLoader
from dataset.coco_dataset import COCOCustomDataset, get_transforms
from models.mlp import MLP
from models.cnn import CNNModel
from models.vision_transformer import VisionTransformer
from models.transformer_encoder import TransformerEncoderModel
from training.train import train_one_epoch, validate_one_epoch
from training.train_utils import collate_fn

def main():
    # Hyperparameters & config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    batch_size = 16
    learning_rate = 1e-4

    # Dataset & DataLoader
    transforms = get_transforms()
    dataset = COCOCustomDataset(transform=transforms)
    n_val = int(0.15 * len(dataset))
    n_test = int(0.15 * len(dataset))
    n_train = len(dataset) - n_val - n_test

    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, 
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, 
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    # -----------------------------------------------------
    # 1) MLP
    mlp_model = MLP(input_dim=224*224*3).to(device)
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    print("\n[Training: MLP Model]")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(mlp_model, mlp_optimizer, train_loader, device=device)
        val_loss, val_metrics = validate_one_epoch(mlp_model, val_loader, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val(MSE={val_metrics['mse']:.2f}, "
              f"MAE={val_metrics['mae']:.2f}, R2={val_metrics['r2']:.2f})")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp_model.state_dict(), "best_mlp.pth")

    # -----------------------------------------------------
    # 2) CNN
    cnn_model = CNNModel().to(device)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    print("\n[Training: CNN Model]")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(cnn_model, cnn_optimizer, train_loader, device=device)
        val_loss, val_metrics = validate_one_epoch(cnn_model, val_loader, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val(MSE={val_metrics['mse']:.2f}, "
              f"MAE={val_metrics['mae']:.2f}, R2={val_metrics['r2']:.2f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(cnn_model.state_dict(), "best_cnn.pth")

    # -----------------------------------------------------
    # 3) Vision Transformer
    vit_model = VisionTransformer().to(device)
    vit_optimizer = torch.optim.Adam(vit_model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    print("\n[Training: Vision Transformer Model]")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(vit_model, vit_optimizer, train_loader, device=device)
        val_loss, val_metrics = validate_one_epoch(vit_model, val_loader, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val(MSE={val_metrics['mse']:.2f}, "
              f"MAE={val_metrics['mae']:.2f}, R2={val_metrics['r2']:.2f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vit_model.state_dict(), "best_vit.pth")

    # -----------------------------------------------------
    # 4) Transformer Encoder
    trans_enc_model = TransformerEncoderModel().to(device)
    trans_enc_optimizer = torch.optim.Adam(trans_enc_model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")

    print("\n[Training: Transformer Encoder Model]")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(trans_enc_model, trans_enc_optimizer, train_loader, device=device)
        val_loss, val_metrics = validate_one_epoch(trans_enc_model, val_loader, device=device)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val(MSE={val_metrics['mse']:.2f}, "
              f"MAE={val_metrics['mae']:.2f}, R2={val_metrics['r2']:.2f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(trans_enc_model.state_dict(), "best_transformer_encoder.pth")

    # -----------------------------------------------------
    trans_enc_model.load_state_dict(torch.load("best_transformer_encoder.pth"))
    test_loss, test_metrics = validate_one_epoch(trans_enc_model, test_loader, device=device)
    print("\n[Final Test - Transformer Encoder]")
    print(f"Test Loss: {test_loss:.4f}  Test(MSE={test_metrics['mse']:.2f}, "
          f"MAE={test_metrics['mae']:.2f}, R2={test_metrics['r2']:.2f})")

if __name__ == "__main__":
    main()
