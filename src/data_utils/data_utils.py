import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
