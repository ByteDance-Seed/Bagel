import os
import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.optimizer import Optimizer

def save_checkpoint(
    model: FSDP,
    ema_model: Optional[FSDP],
    checkpoint_dir: str,
    step: int,
    logger: logging.Logger,
    save_optimizer_state: bool = True,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    data_status: Optional[dict] = None,
):
    """
    Saves a distributed checkpoint for both the main model and the GPU EMA model.

    Args:
        model (FSDP): The main FSDP-wrapped model.
        ema_model (Optional[FSDP]): The FSDP-wrapped EMA model.
        checkpoint_dir (str): The directory to save the checkpoint to.
        step (int): The current training step.
        logger (logging.Logger): The logger for outputting information.
        save_optimizer_state (bool): If True, saves optimizer, scheduler, etc.
        optimizer (Optional[Optimizer]): The optimizer.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The LR scheduler.
        data_status (Optional[dict]): The dataloader status dictionary.
    """
    if save_optimizer_state and (optimizer is None or scheduler is None):
        raise ValueError("Optimizer and scheduler must be provided when saving full training state.")

    if dist.get_rank() == 0:
        logger.info(f"Starting to save distributed checkpoint for step {step}...")

    # Create the directory for the specific checkpoint step
    step_checkpoint_dir = os.path.join(checkpoint_dir, f"step_{step}")
    
    # Prepare the state dictionary. This will contain all components to be saved.
    state_dict = {"model": model.state_dict()}

    if ema_model is not None:
        state_dict["ema_model"] = ema_model.state_dict()

    if save_optimizer_state:
        state_dict["optimizer"] = optimizer.state_dict()
    
    # Use DCP to save the entire state dict.
    # DCP will handle sharding everything correctly.
    dcp.save(
        state_dict=state_dict,
        checkpoint_id=step_checkpoint_dir,
    )

    # Rank 0 is responsible for saving non-distributed metadata
    if dist.get_rank() == 0:
        if save_optimizer_state:
            # Save metadata like the current training step
            metadata = {"train_step": step}
            torch.save(metadata, os.path.join(step_checkpoint_dir, "metadata.pt"))
            
            # Save scheduler state
            torch.save(scheduler.state_dict(), os.path.join(step_checkpoint_dir, "scheduler.pt"))
            
            # Save dataloader status if provided
            if data_status:
                torch.save(data_status, os.path.join(step_checkpoint_dir, "data_status.pt"))
        
        logger.info(f"Checkpoint successfully saved to {step_checkpoint_dir}")

    dist.barrier()


def load_checkpoint(
    model: FSDP,
    ema_model: Optional[FSDP],
    resume_from: str,
    logger: logging.Logger,
    resume_model_only: bool = False,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """
    Loads a distributed checkpoint into the FSDP model, EMA model, and optionally the optimizer/scheduler.

    Args:
        model (FSDP): The main FSDP-wrapped model.
        ema_model (Optional[FSDP]): The FSDP-wrapped EMA model.
        resume_from (str): The path to the checkpoint directory (e.g., '.../step_1000').
        logger (logging.Logger): The logger for outputting information.
        resume_model_only (bool): If True, only loads model weights.
        optimizer (Optional[Optimizer]): The optimizer to load state into.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The scheduler to load state into.

    Returns:
        A tuple containing: (train_step, data_status)
    """
    if not resume_model_only and (optimizer is None or scheduler is None):
        raise ValueError("Optimizer and scheduler must be provided when resuming full training state.")

    logger.info(f"Loading distributed checkpoint from {resume_from}.")
    
    # Prepare the state_dict objects that will be populated by dcp.load()
    state_dict_to_load = {"model": model.state_dict()}
    
    if ema_model is not None:
        # It's good practice to check if the ema_model weights will be present in the checkpoint.
        # Here we assume if an ema_model is passed, we intend to load it.
        state_dict_to_load["ema_model"] = ema_model.state_dict()
    
    if not resume_model_only:
        state_dict_to_load["optimizer"] = optimizer.state_dict()
    
    # DCP loads the state from the checkpoint files into the provided state_dict objects.
    # It handles resharding automatically if the world size has changed.
    dcp.load(
        state_dict=state_dict_to_load,
        checkpoint_id=resume_from,
    )

    # --- Load state into each component ---
    model.load_state_dict(state_dict_to_load["model"], assign=True, strict=False)
    logger.info("Successfully loaded main model state dict.")

    if ema_model is not None:
        if "ema_model" in state_dict_to_load:
            ema_model.load_state_dict(state_dict_to_load["ema_model"], assign=True, strict=False)
            logger.info("Successfully loaded EMA model state dict.")
        else:
            logger.warning("EMA model was provided, but 'ema_model' key not found in checkpoint. EMA model is not loaded.")

    # Initialize return values
    train_step = 0
    data_status = None

    if resume_model_only:
        logger.info("Loaded model weights only. Optimizer and scheduler states were not loaded.")
    else:
        optimizer.load_state_dict(state_dict_to_load["optimizer"])
        logger.info("Successfully loaded optimizer state dict.")

        # Load metadata and scheduler state from rank 0's files
        scheduler_path = os.path.join(resume_from, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            logger.info("Successfully loaded scheduler state dict.")
        else:
            logger.warning("Scheduler state dict not found. Starting with a fresh scheduler.")

        metadata_path = os.path.join(resume_from, "metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location="cpu")
            # The checkpoint was saved AT `step`, so we will resume training AT `step`.
            train_step = metadata.get("train_step", 0)
            logger.info(f"Loaded train_step {train_step} from metadata.")
        else:
            logger.warning("metadata.pt not found. Resetting train_step to 0.")
            train_step = 0

        data_status_path = os.path.join(resume_from, "data_status.pt")
        if os.path.exists(data_status_path):
            data_status = torch.load(data_status_path, map_location="cpu")
            logger.info("Successfully loaded data status.")
        else:
            data_status = None
            logger.info("Data status file not found.")

    dist.barrier()
    return train_step, data_status