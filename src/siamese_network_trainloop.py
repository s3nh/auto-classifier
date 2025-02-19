import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from datetime import datetime
import logging
import wandb  # Optional: for experiment tracking
from pathlib import Path
import json

class TrainingConfig:
    def __init__(self):
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.num_epochs = 50
        self.early_stopping_patience = 5
        self.weight_decay = 1e-5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path('checkpoints')
        self.save_dir.mkdir(exist_ok=True)
        
        # Logging configuration
        self.log_interval = 100
        self.eval_interval = 500
        
        # Current timestamp for run identification
        self.timestamp = "2025-02-19_09:52:34"
        self.run_name = f"siamese_fraud_{self.timestamp}"

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop

def train_siamese_network(
    model,
    train_loader,
    val_loader,
    config: TrainingConfig,
    user="s3nh"
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Setup logging
    fh = logging.FileHandler(f'training_{config.timestamp}.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    logger.info(f"Training started by user {user} at {config.timestamp}")
    logger.info(f"Using device: {config.device}")
    
    # Initialize wandb for experiment tracking (optional)
    wandb.init(
        project="fraud-detection-siamese",
        name=config.run_name,
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.num_epochs,
            "weight_decay": config.weight_decay
        }
    )
    
    model = model.to(config.device)
    criterion = nn.ContrastiveLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3
    )
    
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    # Training metrics
    best_val_loss = float('inf')
    global_step = 0
    training_metrics = []
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        batch_metrics = []
        
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            x1 = x1.to(config.device)
            x2 = x2.to(config.device)
            labels = labels.to(config.device)
            
            # Forward pass
            output1, output2 = model(x1, x2)
            loss = criterion(output1, output2, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                distances = nn.functional.pairwise_distance(output1, output2)
                predictions = (distances < 0.5).float()
                accuracy = (predictions == labels).float().mean()
            
            # Log metrics
            batch_metrics.append({
                'loss': loss.item(),
                'accuracy': accuracy.item()
            })
            epoch_loss += loss.item()
            
            if batch_idx % config.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch}/{config.num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Accuracy: {accuracy.item():.4f}"
                )
                
                wandb.log({
                    'train_loss': loss.item(),
                    'train_accuracy': accuracy.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'global_step': global_step
                })
            
            # Validation
            if batch_idx % config.eval_interval == 0:
                val_metrics = validate(
                    model, 
                    val_loader, 
                    criterion, 
                    config.device
                )
                
                logger.info(
                    f"Validation - Loss: {val_metrics['val_loss']:.4f} "
                    f"AUC: {val_metrics['val_auc']:.4f} "
                    f"AP: {val_metrics['val_ap']:.4f}"
                )
                
                wandb.log({
                    'val_loss': val_metrics['val_loss'],
                    'val_auc': val_metrics['val_auc'],
                    'val_ap': val_metrics['val_ap'],
                    'global_step': global_step
                })
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        val_metrics,
                        config.save_dir / f"best_model_{config.timestamp}.pt"
                    )
                
                # Early stopping check
                if early_stopping(val_metrics['val_loss']):
                    logger.info("Early stopping triggered")
                    break
                
                model.train()
            
            global_step += 1
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_epoch_loss)
        
        # Save epoch metrics
        training_metrics.append({
            'epoch': epoch,
            'avg_loss': avg_epoch_loss,
            'batch_metrics': batch_metrics
        })
        
        # Save training metrics to file
        with open(f'training_metrics_{config.timestamp}.json', 'w') as f:
            json.dump(training_metrics, f)
    
    logger.info("Training completed")
    wandb.finish()
    return model, training_metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_labels = []
    all_distances = []
    
    with torch.no_grad():
        for x1, x2, labels in val_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            
            output1, output2 = model(x1, x2)
            loss = criterion(output1, output2, labels)
            distances = nn.functional.pairwise_distance(output1, output2)
            
            val_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_distances.extend(distances.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    all_labels = np.array(all_labels)
    all_distances = np.array(all_distances)
    
    # Calculate metrics
    val_auc = roc_auc_score(all_labels, -all_distances)
    val_ap = average_precision_score(all_labels, -all_distances)
    
    return {
        'val_loss': avg_val_loss,
        'val_auc': val_auc,
        'val_ap': val_ap
    }

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)

# Example usage
if __name__ == "__main__":
    # Initialize your model, datasets, and config
    config = TrainingConfig()
    
    # Create your model and data loaders
    model = SiameseNetwork(input_dim=20)  # Adjust input_dim based on your data
    
    # Example data loaders (replace with your actual data)
    train_loader = DataLoader(
        your_train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        your_val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Train the model
    trained_model, metrics = train_siamese_network(
        model,
        train_loader,
        val_loader,
        config
    )
