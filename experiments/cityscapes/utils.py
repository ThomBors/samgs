import numpy as np
import torch
import logging

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).cpu().numpy(), acc.cpu().numpy()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (
        torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item(), (
        torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item()


# for calculating \Delta_m
delta_stats = [
    "mean iou",
    "pix acc",
    "abs err",
    "rel err",
]
BASE = np.array(
    #[0.3830, 0.6376, 0.6754, 0.2780, 25.01, 19.21, 0.3014, 0.5720, 0.6915]
    [0.7401, 0.9316, 0.0125, 27.77]
)  # base results from CAGrad
SIGN = np.array([1, 1, 0, 0])
KK = np.ones(4) * -1


def delta_fn(a):
    return (KK ** SIGN * (a - BASE) / BASE).mean() * 100.0  # * 100 for percentage



def load_checkpoint(chk_path, model, optimizer, scheduler, device, epoch=None):
    
    checkpoint_files = sorted(chk_path.glob("chk_epoch_*.pth"), key=lambda f: int(f.stem.split('_')[2]), reverse=True)
    checkpoint_file = checkpoint_files[0]
    epoch = int(checkpoint_file.stem.split('_')[2])
    checkpoint_file = chk_path / f"chk_epoch_{epoch}.pth"

    previous_epoch = epoch - 1
    previous_checkpoint_file = chk_path / f"chk_epoch_{previous_epoch}.pth"
    
    try:
        checkpoint = torch.load(previous_checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        latest_epoch = checkpoint['epoch']
        custom_step = checkpoint['custom_step']
        deltas = checkpoint['metrics']['delta_m']
        keys = checkpoint['metrics']['keys']
        avg_cost = checkpoint['metrics']['avg_cost']
        loss_list = checkpoint['metrics']['losses']

        logging.info(f"Resumed from checkpoint: {previous_checkpoint_file}, epoch {latest_epoch}")
        return latest_epoch, custom_step, deltas, keys, avg_cost, loss_list
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {previous_checkpoint_file}: {e}")
        raise RuntimeError(f"Failed to load checkpoint {previous_checkpoint_file}.")


def save_checkpoint(chk_path, model, optimizer, scheduler, epoch, custom_step, deltas, keys, avg_cost, loss_list):
    # Define file path for checkpoint `epoch`
    save_path = chk_path / f"chk_epoch_{epoch}.pth"

    # Save the checkpoint for the current epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'custom_step': custom_step,
        'metrics': {
            "delta_m": deltas,
            "keys": keys,
            "avg_cost": avg_cost,
            "losses": loss_list,
        }
    }, save_path)
    logging.info(f"Checkpoint saved to: {save_path}")