import argparse
import torch

from mlp_mixer import MlpMixer
from dataloaders import get_dataloaders
from utils import LabelSmoothingLoss, get_optimizer, WarmupCosineLR, AverageMeter, cutmix_data


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP-Mixer on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--augmentation', type=str, default='autoaugment', choices=['autoaugment', 'randaugment'],
                        help='Type of data augmentation to use')
    return parser.parse_args()


# Training function
def train(model, trainloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply CutMix
        inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss_meter.update(loss.item(), inputs.size(0))
        accuracy = (outputs.argmax(dim=-1) == labels).sum().item() / inputs.size(0)
        train_acc_meter.update(accuracy, inputs.size(0))

    if device.type == 'cuda':
        allocated_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserved_memory = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(f'[Epoch {epoch + 1}] -- Train Loss: {train_loss_meter.average:.3f} -- Train Accuracy: {train_acc_meter.average:.3f} -- Allocated Memory: {allocated_memory:.2f} GB -- Reserved Memory: {reserved_memory:.2f} GB')
    else:
        print(f'[Epoch {epoch + 1}] -- Train Loss: {train_loss_meter.average:.3f} -- Train Accuracy: {train_acc_meter.average:.3f}')
    scheduler.epoch_step()


# Testing function
def test(model, testloader, criterion, device):
    model.eval()
    test_loss_meter = AverageMeter()
    test_acc_meter = AverageMeter()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss_meter.update(loss.item(), inputs.size(0))
            accuracy = (outputs.argmax(dim=-1) == labels).sum().item() / inputs.size(0)
            test_acc_meter.update(accuracy, inputs.size(0))

    print(f'Test Loss: {test_loss_meter.average:.3f} -- Test Accuracy: {test_acc_meter.average:.3f}')


if __name__ == "__main__":
    args = parse_args()
    print('Training MLP-Mixer on CIFAR-10 with the following configurations:')
    for arg in vars(args):
        print(f'\t{arg}: {getattr(args, arg)}')

    trainloader, testloader = get_dataloaders(args.batch_size, args.num_workers, args.augmentation)

    model = MlpMixer(
        image_shape=(3, 32, 32),
        patch_size=4,
        num_classes=10,
        num_mixers=8,
        num_features=256,
        hidden_dim_token=64,
        hidden_dim_channel=512,
        dropout=0.0
    )
    print(model)
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = LabelSmoothingLoss(smoothing=0.1)
    optimizer = get_optimizer(model, args.lr, args.weight_decay)
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=5, total_epochs=args.epochs, num_batches_per_epoch=len(trainloader), min_lr=1e-6
    )

    for epoch in range(args.epochs):
        train(model, trainloader, criterion, optimizer, scheduler, device, epoch)
        test(model, testloader, criterion, device)

    print('Finished Training')