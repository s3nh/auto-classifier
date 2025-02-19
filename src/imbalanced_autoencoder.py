import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List, Union
import logging
from datetime import datetime
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImbalancedAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        latent_dim: int = 8,
        dropout_rate: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Autoencoder architecture optimized for imbalanced numerical data.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden dimensions for encoder/decoder
            latent_dim: Dimension of the latent space
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(ImbalancedAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32, 16]
        
        # Get activation function
        self.activation = self._get_activation(activation)
        
        # Build encoder
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
            
        # Latent space projection
        encoder_layers.extend([
            nn.Linear(self.hidden_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        current_dim = latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
            
        # Output layer
        decoder_layers.extend([
            nn.Linear(self.hidden_dims[0], input_dim),
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"Created autoencoder with architecture: {input_dim} -> {self.hidden_dims} -> {latent_dim}")
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
            
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for handling imbalanced data."""
    def __init__(self, weight_dict: dict = None):
        super(WeightedMSELoss, self).__init__()
        self.weight_dict = weight_dict or {}
        
    def forward(self, x_recon: torch.Tensor, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((x_recon - x) ** 2, dim=1)
        
        if self.weight_dict:
            weights = torch.tensor([self.weight_dict.get(label.item(), 1.0) 
                                  for label in labels]).to(x.device)
            return torch.mean(weights * mse)
        return torch.mean(mse)

class ImbalancedDataset(Dataset):
    """Dataset class for handling imbalanced numerical data."""
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        scaler: Optional[StandardScaler] = None
    ):
        if scaler is None:
            scaler = StandardScaler()
            self.features = scaler.fit_transform(features)
        else:
            self.features = scaler.transform(features)
            
        self.labels = labels
        self.scaler = scaler
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.features[idx]),
                torch.LongTensor([self.labels[idx]]))

def train_autoencoder(
    model: ImbalancedAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience: int = 10,
    user: str = "s3nh",
    timestamp: str = "2025-02-19 10:44:53"
):
    """Train the autoencoder."""
    logger.info(f"Starting training at {timestamp} by user {user}")
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    
    # Initialize wandb
    wandb.init(
        project="imbalanced-autoencoder",
        name=f"training_{timestamp}",
        config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "architecture": str(model)
        }
    )
    
    # Calculate class weights
    label_counts = torch.bincount(torch.tensor([label.item() for _, label in train_loader.dataset]))
    weight_dict = {i: 1.0 / count.item() for i, count in enumerate(label_counts)}
    
    criterion = WeightedMSELoss(weight_dict)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
                
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        reconstruction_errors = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data, labels)
                val_loss += loss.item()
                
                # Calculate reconstruction error per sample
                errors = torch.mean((reconstructed - data) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Log metrics
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })
        
        logger.info(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, f'best_model_{timestamp}.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logger.info("Early stopping triggered")
            break
            
    wandb.finish()
    return model

def get_reconstruction_threshold(
    model: ImbalancedAutoencoder,
    val_loader: DataLoader,
    percentile: float = 95,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """Calculate reconstruction error threshold for anomaly detection."""
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            reconstructed, _ = model(data)
            errors = torch.mean((reconstructed - data) ** 2, dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
            
    threshold = np.percentile(reconstruction_errors, percentile)
    return threshold

# Example usage
if __name__ == "__main__":
    # Generate dummy imbalanced data
    np.random.seed(42)
    n_samples_majority = 1000
    n_samples_minority = 100
    n_features = 70
    
    # Create majority and minority class samples
    X_majority = np.random.randn(n_samples_majority, n_features)
    X_minority = np.random.randn(n_samples_minority, n_features) + 2
    
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([np.zeros(n_samples_majority), np.ones(n_samples_minority)])
    
    # Create datasets
    dataset = ImbalancedDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with weighted sampling for training
    labels = y[train_dataset.indices]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = ImbalancedAutoencoder(
        input_dim=n_features,
        hidden_dims=[56, 32, 16],
        latent_dim=8
    )
    
    trained_model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        learning_rate=1e-3,
        user="s3nh",
        timestamp="2025-02-19 10:44:53"
    )
