from imodels import imodel1, imodel2, imodel3, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from utils import load_data, ToTensor, Compose, RandomHorizontalFlip, ResizeImage, ColorJitter, CustomCrop
import torch.nn.functional as F
import time

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if has_gpu else "cpu"
print("Using ", device)
# torch.backends.cudnn.benchmark = True


def train(args):
    from os import path

    model1_input_size, model1_cut = [144, 192], [0, 300, 0, 400]
    model2_input_size, model2_cut = [128, 288], [92, 284, 0, 400]
    model3_input_size, model3_cut = [144, 192], [0, 300, 0, 400]

    model_input_size = model3_input_size
    model_input_cut = model3_cut
    model = imodel3().to(device)
    train_logger, val_logger = None, None
    acc_calculator = AccuracyCalculator()
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"))
        val_logger = tb.SummaryWriter(path.join(args.log_dir, "val"))

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), "imodel.th")))

    print("Model size: ", measure_model_size(model))
    print("Inference time: ", measure_inference_time(model, torch.Tensor(np.random.rand(2, 3, model_input_size[0], model_input_size[1]).astype(np.float32))))

    model.eval()
    print(model(torch.Tensor(np.random.rand(2, 3, model_input_size[0], model_input_size[1]).astype(np.float32)).to(device)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    backbone_parameters = set(param for param in model.backbone.parameters())
    c_parameters = set(param for param in model.c_part.parameters())
    d_parameters = set(param for param in model.d_part.parameters())
    f_parameters = (
        set(param for param in model.f_f4.parameters())
        | set(param for param in model.f_f5.parameters())
        | set(param for param in model.f_f6.parameters())
        | d_parameters
    )
    reg_parameters = [param for param in model.parameters() if param not in f_parameters]

    reg_optimizer = torch.optim.Adam(reg_parameters, lr=args.learning_rate, weight_decay=1e-5)
    class_optimizer = torch.optim.Adam(f_parameters, lr=args.learning_rate * 5, weight_decay=1e-5)

    augmentation = Compose(
        [
            ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.002),
            RandomHorizontalFlip(0.5),
            # CustomCrop(start_row=model_input_cut[0], end_row=model_input_cut[1]),
            ResizeImage((model_input_size[0], model_input_size[1])),
            ToTensor(),
        ]
    )
    val_augmentations = Compose(
        [
            CustomCrop(start_row=model_input_cut[0], end_row=model_input_cut[1]),
            ResizeImage((model_input_size[0], model_input_size[1])),
            ToTensor(),
        ]
    )
    save_model(model, "tmodel.th")

    loss_center = masked_mse_loss  # torch.nn.MSELoss()
    loss_depth = masked_depth_mse_loss  # torch.nn.MSELoss()
    loss_flag = torch.nn.BCEWithLogitsLoss()
    loss_comparision = torch.nn.MSELoss(reduction="none").to(device)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate * 4, steps_per_epoch=len(train_data), epochs=args.num_epoch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(reg_optimizer, mode="min", factor=1 / 3, patience=5, threshold=0.005, min_lr=1e-5)
    class_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(class_optimizer, mode="min", factor=1 / 3, patience=5, threshold=0.005, min_lr=1e-5)

    global_step = 0
    if True:
        train_data = load_data("/Users/andreas/Movies/train_set", transform=augmentation, num_workers=4, batch_size=224, include_nans=False)
        val_data = load_data("/Users/andreas/Movies/val_set", transform=val_augmentations, num_workers=2, batch_size=32, include_nans=False)

        for epoch in range(args.num_epoch):
            model.train()
            start = time.time()

            for batch, (x, c, d, f) in enumerate(train_data):
                x, c, d, f = x.to(device), c.to(device), d.to(device), f.to(device)
                c_pred, d_pred, f_pred = model(x)

                loss_val1 = loss_center(c_pred, c)
                loss_val2 = loss_depth(d_pred, d)
                loss_val = loss_val1 + loss_val2 * 0.0001

                reg_optimizer.zero_grad()
                loss_val.backward()
                reg_optimizer.step()
                if train_logger is not None:
                    train_logger.add_scalar("center_loss", loss_val1, global_step)
                    train_logger.add_scalar("depth_loss", loss_val2, global_step)
                    train_logger.add_scalar("train_loss", loss_val, global_step)
                global_step += 1
            itime = time.time() - start

            model.eval()
            val_running_loss1 = 0.0
            val_running_loss2 = 0.0
            val_running_loss3 = 0.0

            val_count = 0
            val_count_non_nan = 0
            with torch.no_grad():
                for batch, (x, c, d, f) in enumerate(val_data):
                    x, c, d, f = x.to(device), c.to(device), d.to(device), f.to(device)
                    c_pred, d_pred, f_pred = model(x)

                    loss_val1 = loss_center(c_pred, c)
                    loss_val2 = loss_depth(d_pred, d)
                    loss_val = loss_val1 + loss_val2

                    val_running_loss1 += loss_val1.item() * torch.sum(~torch.isnan(x)).item()
                    val_running_loss2 += loss_val2.item() * torch.sum(~torch.isnan(x)).item()
                    val_running_loss3 += loss_comparision(c_pred, c).mean().item()
                    val_count += x.size(0)
                    val_count_non_nan += torch.sum(~torch.isnan(x))

            val_epoch_loss1 = val_running_loss1 / val_count_non_nan
            val_epoch_loss2 = val_running_loss2 / val_count_non_nan
            val_epoch_loss3 = val_running_loss3 / batch

            val_epoch_loss = val_epoch_loss1 + val_epoch_loss2
            print(f"{epoch} - val_epoch_loss: ", val_epoch_loss, "  time:", itime, "  ", val_epoch_loss1, "  ", val_epoch_loss2, "  ", val_epoch_loss3, batch)
            scheduler.step(val_epoch_loss)
            save_model(model, "imodel2_128_288m.th")

            if val_logger is not None:
                val_logger.add_scalar("val_center_loss", val_epoch_loss1, global_step)
                val_logger.add_scalar("val_depth_loss", val_epoch_loss2, global_step)
                val_logger.add_scalar("val_loss", val_epoch_loss, global_step)

            if True:  # save_example
                save_example(x, c, c_pred, global_step)

    last_reg_step = global_step

    for param in reg_parameters:
        param.requires_grad = False
    if True:
        train_data = load_data("/root/dev/data/train_set/train_set", transform=augmentation, num_workers=4, batch_size=224, include_nans=True)
        val_data = load_data("/root/dev/data/val_set/val_set", transform=val_augmentations, num_workers=2, batch_size=32, include_nans=True)
        for epoch in range(args.num_epoch):
            model.train()
            start = time.time()

            for batch, (x, c, d, f) in enumerate(train_data):
                x, c, d, f = x.to(device), c.to(device), d.to(device), f.to(device)
                c_pred, d_pred, f_pred = model(x)

                loss_val = loss_flag(f_pred, f)
                acc_calculator.add(f_pred[:, 0], f[:, 0])
                class_optimizer.zero_grad()
                loss_val.backward()
                class_optimizer.step()
                if train_logger is not None:
                    train_logger.add_scalar("flag_loss", loss_val, global_step - last_reg_step)
                global_step += 1
            itime = time.time() - start
            train_acc_epoch = acc_calculator.accuracy
            if train_logger is not None:
                train_logger.add_scalar("flag_acc", train_acc_epoch, global_step - last_reg_step)
            acc_calculator.reset()

            model.eval()
            val_running_loss1 = 0.0

            val_count = 0
            val_count_non_nan = 0
            with torch.no_grad():
                for batch, (x, c, d, f) in enumerate(val_data):
                    x, c, d, f = x.to(device), c.to(device), d.to(device), f.to(device)
                    c_pred, d_pred, f_pred = model(x)

                    loss_val = loss_flag(f_pred, f)
                    acc_calculator.add(f_pred[:, 0], f[:, 0])

                    val_running_loss1 += loss_val.item() * torch.sum(~torch.isnan(x)).item()
                    val_count += x.size(0)
                    val_count_non_nan += torch.sum(~torch.isnan(x))

            val_epoch_loss1 = val_running_loss1 / val_count
            acc_epoch = acc_calculator.accuracy
            print(
                f"{epoch} - val_epoch_loss: ",
                val_epoch_loss1,
                "  time:",
                itime,
                "  ",
                "accuracy: ",
                acc_epoch,
                "  old:",
                val_epoch_loss2,
                "  ",
                val_epoch_loss3,
            )
            if val_logger is not None:
                val_logger.add_scalar("val_flag_acc", acc_epoch, global_step - last_reg_step)
                val_logger.add_scalar("val_flag_loss", val_epoch_loss1, global_step - last_reg_step)
            class_scheduler.step(loss_val)
            acc_calculator.reset()
            save_model(model, "imodel2_128_288.th")
    scripted_model = torch.jit.script(model.to("cpu"))
    torch.jit.save(scripted_model, "imodel_jit_cpu2.pt")


def save_example(img, label, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    import os

    save_dir = "/root/dev/data/example_outputs"
    save_name = os.path.join(save_dir, str(global_step) + ".png")
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy()[::-1] + 1), 2, ec="g", fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy()[::-1] + 1), 2, ec="r", fill=False, lw=1.5))
    plt.savefig(save_name)
    plt.close(fig)
    del ax, fig


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec="g", fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec="r", fill=False, lw=1.5))
    logger.add_figure("viz", fig, global_step)
    del ax, fig


def masked_mse_loss(c_pred, c):
    mask = ~torch.isnan(c)
    mask = mask.all(dim=1)

    c_pred_valid = c_pred[mask]
    c_valid = c[mask]

    # Calculate MSE Loss only on valid (non-NaN) entries
    if c_pred_valid.nelement() == 0 or c_valid.nelement() == 0:
        return torch.tensor(0.0, device=c_pred.device, requires_grad=True)
    loss = F.mse_loss(c_pred_valid, c_valid)
    return loss


def masked_depth_mse_loss(d_pred, d):
    # Check for NaN values in the target tensor 'c' to create a mask; this keeps dimensions intact
    mask = ~torch.isnan(d).squeeze()

    # Apply the mask to flatten both predicted and target tensors, removing entries with NaN
    d_pred_valid = d_pred.squeeze()[mask]
    d_valid = d.squeeze()[mask]

    # Calculate MSE Loss only on valid (non-NaN) entries
    if d_pred_valid.nelement() == 0 or d_valid.nelement() == 0:
        return torch.tensor(0.0, device=d_pred.device, requires_grad=True)
    loss = F.mse_loss(d_pred_valid, d_valid)
    return loss


def measure_inference_time(model, input):
    num = 100
    if device == "cuda":
        model.cuda()
        input = input.cuda()
        torch.cuda.synchronize()
    elif device == "mps":
        input = input.to(device)
    start = time.time()
    for _ in range(num):
        with torch.no_grad():
            output = model(input)
        if device == "cuda":
            torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000
    return total_time / num


def measure_model_size(model):  # in kb
    import pickle

    pickled_model = pickle.dumps(model.state_dict())
    size_in_kb = len(pickled_model) / 1024
    return size_in_kb


class AccuracyCalculator:
    def __init__(self):
        self.total_correct = 0
        self.total_samples = 0

    def add(self, logits, ground_truth):
        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            correct = (predictions == ground_truth).float()
            self.total_correct += correct.sum().item()
            self.total_samples += ground_truth.size(0)

    def reset(self):
        self.total_correct = 0
        self.total_samples = 0

    @property
    def accuracy(self):
        if self.total_samples == 0:
            return 0  # Avoid division by zero
        return self.total_correct / self.total_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir")
    parser.add_argument("-c", "--continue_training", action="store_true")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-n", "--num_epoch", type=int, default=40)

    # Put custom arguments here

    args = parser.parse_args()
    train(args)
