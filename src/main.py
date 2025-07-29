import torch
from data_utils.data_utils import load_and_preprocess_data, CSVDataset
from torch.utils.data import DataLoader
from model.mlp import MLP
import torch.nn as nn
import torch.optim as optim
from training.training import train_model
from visualization.plot_training import plot_training_curves 


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
