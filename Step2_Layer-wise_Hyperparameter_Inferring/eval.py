import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import argparse
from models import MateModel_Hyper
from dataset import RaplLoader


class F1_score(nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def reset(self):
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.num_classes)
        y_pred = F.one_hot(torch.argmax(y_pred, dim=1), self.num_classes)

        self.tp += (y_true * y_pred).sum(0)
        self.tn += ((1 - y_true) * (1 - y_pred)).sum(0)
        self.fp += ((1 - y_true) * y_pred).sum(0)
        self.fn += (y_true * (1 - y_pred)).sum(0)

        precision = self.tp / (self.tp + self.fp + self.epsilon)
        recall = self.tp / (self.tp + self.fn + self.epsilon)

        accuracy = self.tp.sum() / (self.tp.sum() + self.tn.sum() + self.fp.sum() + self.fn.sum())
        accuracy = accuracy.item() * self.num_classes

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.mean().item()
        return accuracy*100., precision.mean().item()*100., recall.mean().item()*100., f1*100.


@torch.no_grad()
def eval_step(loader):
    net.eval()

    eval_loss, accuracy, F1 = 0, 0, 0
    f1.reset()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        outputs = net(inputs)
        loss = ce(outputs, targets)

        eval_loss += loss.item()
        accuracy, p, r, F1 = f1(outputs, targets)

    logs = 'Loss: {:.3f}\t Acc: {:.3f}\t P: {:.3f}\t R: {:.3f}\t F1: {:.3f}\t'
    print(logs.format(eval_loss / len(loader), accuracy, p, r, F1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepPower Evaluate')
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--path', default='results/model_K_final', type=str, help='load_path')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print('Loading data...')
    data = RaplLoader(batch_size=args.batch_size, num_workers=args.workers, mode='kernel_size')
    dataloader = data.get_loader()

    print('Loading model...')
    net = MateModel_Hyper.Model(num_classes=data.num_classes).to(device)
    checkpoint = torch.load(args.path + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    print("Under evaluation...")
    ce = nn.CrossEntropyLoss()
    f1 = F1_score(num_classes=data.num_classes)
    eval_step(dataloader)
