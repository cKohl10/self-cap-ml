import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=3):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class CSVDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_and_preprocess_data(csv_path, input_cols, output_cols, test_size=0.2, random_state=42):
    """
    Load data from CSV and preprocess it
    
    Args:
        csv_path: Path to CSV file
        input_cols: List of column names for input features
        output_cols: List of column names for output targets
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract features and targets
    X = df[input_cols].values
    y = df[output_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale targets
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    return (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
            scaler_X, scaler_y)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, device='cpu'):
    """
    Train the model
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def plot_training_curves(train_losses, val_losses):
    """
    Plot training and validation loss curves
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Configuration
    csv_path = 'path/to/your/data.csv'  # Update this path
    input_cols = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6']  # Update column names
    output_cols = ['output1', 'output2', 'output3']  # Update column names
    
    # Hyperparameters
    hidden_size = 64
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (X_train, y_train, X_test, y_test, scaler_X, scaler_y) = load_and_preprocess_data(
        csv_path, input_cols, output_cols
    )
    
    # Create datasets and dataloaders
    train_dataset = CSVDataset(X_train, y_train)
    test_dataset = CSVDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MLP(input_size=6, hidden_size=hidden_size, output_size=3).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'input_cols': input_cols,
        'output_cols': output_cols
    }, 'mlp_model.pth')
    
    print("Training completed! Model saved as 'mlp_model.pth'")

if __name__ == "__main__":
    main()
