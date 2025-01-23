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
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
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

    scheduler.epoch_step()

    return train_loss_meter.average, train_acc_meter.average


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

    return test_loss_meter.average, test_acc_meter.average


if __name__ == "__main__":
    import time

    start_time = time.time()

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
        num_features=128,
        hidden_dim_token=64,
        hidden_dim_channel=512,
        dropout=0.0
    )
    print(model)
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad): ,}')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion_train = LabelSmoothingLoss(smoothing=0.1)
    criterion_test = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = WarmupCosineLR(
        optimizer, warmup_epochs=5, total_epochs=args.epochs, num_batches_per_epoch=len(trainloader), min_lr=1e-6
    )
    print(f"Optimizer: {optimizer}")
    print(f"Scheduler: {scheduler}")

    for epoch in range(args.epochs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        train_loss, train_acc = train(model, trainloader, criterion_train, optimizer, scheduler, device, epoch)
        test_loss, test_acc = test(model, testloader, criterion_test, device)
        print(f"Epoch [{epoch + 1}/{args.epochs}] -- Loss: {test_loss:.3f} ({train_loss:.3f}) -- Accuracy: {test_acc*100:.3f}% ({train_acc*100:.3f}%) [lr: {optimizer.param_groups[0]['lr']:.6f}]")
        if device.type == 'cuda':
            print(f"Peak memory allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GB -- Peak memory reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GB")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Finished Training in {elapsed_time:.2f} seconds')
