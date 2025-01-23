"""
# deepseek
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from your_model_file import ValuePolicyNet  # Import your model
from config import config  # Import your config
from your_dataset_file import ChessDataset  # Assume you have a dataset class

# Initialize the model, optimizer, and loss functions
model = ValuePolicyNet(config)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

# Assuming you have a dataset and a data loader
train_dataset = ChessDataset(...)  # You need to define this
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# Training loop
def train(model, train_loader, optimizer, policy_loss_fn, value_loss_fn, epochs, device='cuda'):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        
        for batch_idx, (data, policy_target, value_target) in enumerate(train_loader):
            data, policy_target, value_target = data.to(device), policy_target.to(device), value_target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            policy_pred, value_pred = model(data)
            
            # Compute loss
            policy_loss = policy_loss_fn(policy_pred, policy_target)
            value_loss = value_loss_fn(value_pred, value_target)
            loss = policy_loss + value_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            
            # Logging
            if batch_idx % config.log_interval == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Policy Loss: {policy_loss.item():.4f}, '
                      f'Value Loss: {value_loss.item():.4f}, '
                      f'Total Loss: {loss.item():.4f}')
        
        # Print average losses for the epoch
        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Avg Policy Loss: {avg_policy_loss:.4f}, '
              f'Avg Value Loss: {avg_value_loss:.4f}, '
              f'Avg Total Loss: {avg_loss:.4f}')
    
    # Save the model after training
    model.save_model('value_policy_net.pth')

# Run the training
train(model, train_loader, optimizer, policy_loss_fn, value_loss_fn, epochs=config.epochs)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
import time

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_dataloaders(dataset, batch_size, train_split=0.8):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def validate(model, val_loader, policy_loss_fn, value_loss_fn, device):
    model.eval()
    total_policy_loss = 0
    total_value_loss = 0
    
    with torch.no_grad():
        for data, policy_target, value_target in val_loader:
            data = data.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            
            policy_pred, value_pred = model(data)
            policy_loss = policy_loss_fn(policy_pred, policy_target)
            value_loss = value_loss_fn(value_pred, value_target)
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    
    avg_policy_loss = total_policy_loss / len(val_loader)
    avg_value_loss = total_value_loss / len(val_loader)
    return avg_policy_loss, avg_value_loss

def train_model(config, model, dataset):
    setup_logging()
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    train_loader, val_loader = create_dataloaders(dataset, config.batch_size)
    
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 10
    
    for epoch in range(config.epochs):
        model.train()
        epoch_start_time = time.time()
        total_policy_loss = 0
        total_value_loss = 0
        
        for batch_idx, (data, policy_target, value_target) in enumerate(train_loader):
            data = data.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            
            optimizer.zero_grad()
            policy_pred, value_pred = model(data)
            
            policy_loss = policy_loss_fn(policy_pred, policy_target)
            value_loss = value_loss_fn(value_pred, value_target)
            loss = policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            if batch_idx % config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{config.epochs} "
                    f"[{batch_idx}/{len(train_loader)}] "
                    f"Policy Loss: {policy_loss.item():.4f} "
                    f"Value Loss: {value_loss.item():.4f}"
                )
        
        # Validation
        val_policy_loss, val_value_loss = validate(model, val_loader, policy_loss_fn, value_loss_fn, device)
        val_total_loss = val_policy_loss + val_value_loss
        
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.2f}s. "
            f"Validation Policy Loss: {val_policy_loss:.4f} "
            f"Validation Value Loss: {val_value_loss:.4f}"
        )
        
        # Learning rate scheduling
        scheduler.step(val_total_loss)
        
        # Save checkpoint if best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            early_stopping_counter = 0
            checkpoint_path = checkpoints_dir / f"model_epoch_{epoch+1}_loss_{val_total_loss:.4f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info("Early stopping triggered")
            break
    
    logger.info("Training completed")
    return model

# Usage example:
if __name__ == "__main__":
    from your_model_file import ValuePolicyNet
    from your_dataset_file import ChessDataset
    from config import config
    
    model = ValuePolicyNet(config)
    dataset = ChessDataset(...)  # Initialize your dataset
    trained_model = train_model(config, model, dataset)