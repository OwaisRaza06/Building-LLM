"""
First when training starts a folder should be created named checkpoints in directory of the model where all checkpoint.pth files will be stored, incase if we want to upload weights to an external source we do so accordingly but the logic remains same. Then checkpoints will be created with certain frequency depending on total number of epochs like after every 20 epochs for total 500 epochs or time based like after every 20 mins 

In every epoch we will keeping track of best model using validation set metrics and whenever a model state is encountered where its performing best compared to epochs so far a checkpoint will be created for that model state and later restored after training finishes

We also want a mechanism so when training starts it looks for checkpoints if it doesn't finds any it assumes the training is beginning from start, if it finds checkpoints then loads the most recent checkpoint and continue traning for total_epoch - checkpoint_epoch number of epochs left 

Things to include in checkpointing:
* model state
* optimizer state
* epoch
* random number generator - if any
* best validation loss
* current validation loss

There will be three kinds of checkpoints:
* latest.pth -> current epoch's checkpoint this will be overwritten in every epoch and will contain the most recent checkpoint

* best.pth -> this will be overwritten when highest validation loss so far has been achieved and will contain the best model configuration 

* prev_checkpoints/ -> this directory contains all the checkpoints recorded periodically during training

* All the checkpoints will first be written to a temporary file and then saved to actual file this will prevent corruption of all the checkpoints in case a failure occurs during creation of checkpoint
"""

from pathlib import Path
import tempfile
import torch

def create_checkpoint(sub_dir_name, filename, model, optimizer, epoch, best_val_loss):
    p = Path('checkpoints')
    checkpoint_dir = p/sub_dir_name
    checkpoint.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dicct(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    with tempfile.NamedTemporaryFile(mode='w+b', dir=checkpoint_dir) as temp:
        tempfile_path = Path(temp.name)
        torch.save(checkpoint, tempfile_path)
        tempfile_path.replace(checkpoint_dir/filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state(checkpoint['optimizer_state'])
    epoch = checkpoint['epoch']+1
    best_val_loss = checkpoint['best_val_loss']
    return epoch, best_val_loss