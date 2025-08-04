import logging
import os
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('text_to_video')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_model_info(logger, model, config):
    """Log model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created successfully")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model configuration: {config}")


def log_training_info(logger, train_loader, val_loader, config):
    """Log training information."""
    logger.info(f"Training setup completed")
    logger.info(f"Training batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Training configuration: {config}")


def log_epoch_info(logger, epoch, train_loss, val_loss=None, lr=None):
    """Log epoch information."""
    log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f}"
    if val_loss is not None:
        log_msg += f", Val Loss: {val_loss:.4f}"
    if lr is not None:
        log_msg += f", LR: {lr:.6f}"
    
    logger.info(log_msg) 