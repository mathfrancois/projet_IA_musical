import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. DATASET (Votre code) ---
class SingingVoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.x_dir = os.path.join(root_dir, 'X')
        self.y_dir = os.path.join(root_dir, 'Y')
        self.files = [f for f in os.listdir(self.x_dir) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        x_spec = np.load(os.path.join(self.x_dir, filename))
        y_spec = np.load(os.path.join(self.y_dir, filename))
        
        # PyTorch attend (Batch, Channel, Height, Width)
        # Nos .npy sont (80, 128) -> On ajoute la dimension Channel (1, 80, 128)
        x_tensor = torch.from_numpy(x_spec).unsqueeze(0)
        y_tensor = torch.from_numpy(y_spec).unsqueeze(0)
        
        return x_tensor, y_tensor

# --- 2. MODÈLE (Simple U-Net) ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # --- Encoder (Downsampling) ---
        # Entrée: (1, 80, 128)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # -> (32, 40, 64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # -> (64, 20, 32)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # -> (128, 10, 16)
        )
        
        # --- Bottleneck (Le "Cerveau" central) ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # --- Decoder (Upsampling) ---
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 256 car concat (128+128)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # 128 car concat (64+64)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # 64 car concat (32+32)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # --- Sortie ---
        self.final = nn.Conv2d(32, 1, kernel_size=1) # Retour à 1 canal (Grayscale)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder + Skip Connections
        # On concatène la sortie de l'encoder correspondant pour garder les détails
        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1) 
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

# --- 3. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16    # Ajustez selon votre VRAM (essayez 32 ou 64 si vous avez un GPU)
LEARNING_RATE = 0.001
EPOCHS = 10         # Pour tester rapidement. Mettez 50 ou 100 pour un vrai résultat.
DATASET_PATH = "./Dataset_AI_Ready" # Le dossier créé précédemment

def main():
    print(f"Utilisation du device : {DEVICE}")
    
    # 1. Chargement des données
    train_dataset = SingingVoiceDataset(os.path.join(DATASET_PATH, 'train'))
    val_dataset = SingingVoiceDataset(os.path.join(DATASET_PATH, 'val'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train size: {len(train_dataset)} segments | Val size: {len(val_dataset)} segments")
    
    # 2. Init Modèle
    model = SimpleUNet().to(DEVICE)
    
    # 3. Loss et Optimizer
    # MSELoss (Mean Squared Error) est standard pour la reconstruction d'image (spectrogramme)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 4. BOUCLE D'ENTRAÎNEMENT ---
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Barre de progression
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Sauvegarde du modèle (checkpoint)
        torch.save(model.state_dict(), "model_checkpoint.pth")

    print("Entraînement terminé !")
    
    # Plot de la courbe d'apprentissage
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Courbe d'apprentissage")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("training_curve.png")
    print("Graphique sauvegardé sous 'training_curve.png'")

if __name__ == "__main__":
    main()