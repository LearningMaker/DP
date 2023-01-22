import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
from models import MateModel_Stru
from dataset import RaplLoader
import Levenshtein


def levenshtein_accuracy(predict, targets, labels):
    def decode(result_not_allign):
        _, indices = result_not_allign.max(-1)
        model_out = indices.tolist()

        decoded = [model_out[0]]
        prev_element = model_out[0]
        for idx in range(1, len(model_out)):
            if prev_element != model_out[idx]:
                decoded.append(model_out[idx])
            prev_element = model_out[idx]

        return decoded

    str_predict, str_labels = [], []
    for (p, t, l) in zip(predict.permute(1, 0, 2), targets, labels):
        p = p[t >= 0, :]
        decoded = decode(p)
        str_predict.append(''.join(str(i) for i in decoded))
        str_labels.append(''.join(str(i) for i in l.tolist()))

    errs = [Levenshtein.distance(str_predict[i], str_labels[i]) for i in range(len(str_predict))]
    tokens = [max(len(str_predict[i]), len(str_labels[i])) for i in range(len(str_predict))]
    LDA = [1 - errs[i] / tokens[i] for i in range(len(str_predict))]
    return LDA


def LDA(out, targets, position):
    out = out.permute(2, 0, 1).log_softmax(-1)

    labels = []
    for (t, pos) in zip(targets, position):
        t = t[t >= 0]
        labels.append(t[pos[:, 0]])

    LDA = levenshtein_accuracy(out, targets, labels)
    return LDA


def UPloss(predicts, targets, position):
    predicts = predicts.softmax(1)
    losses = []
    for (p, t, pos) in zip(predicts, targets, position):
        p = p[:, t >= 0]
        t = t[t >= 0]
        loss = p.new_zeros((1,))
        n = 0
        while n < len(pos):
            i, j = pos[n, 0], pos[n, 1]
            loss[0] += p[t[i], i:j+1].mean().log()
            n += 1

        losses.append(-loss[0][None] / n)

    output = torch.cat(losses, 0)
    output = output.mean()
    return output


@torch.no_grad()
def eval_step(loader):
    net.eval()

    eval_ce_loss, eval_ctc_loss = 0, 0
    seg_acc, seg_n = 0, 0
    levenshtein_acc, levenshtein_n = 0, 0
    for batch_idx, (inputs, targets, position) in enumerate(loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        out = net(inputs)
        loss1 = ce(out, targets)
        loss2 = UPloss(out, targets, position)
        eval_ce_loss += loss1.item()
        eval_ctc_loss += loss2.item()

        _, predicted = out.max(1)
        labeled = (targets >= 0)
        total = labeled.sum(1)
        correct = ((predicted == targets) * labeled).sum(1)
        acc = 100. * correct / total
        seg_acc += sum(acc)
        seg_n += len(acc)

        lda = LDA(out, targets, position)
        levenshtein_acc += sum(lda)
        levenshtein_n += len(lda)

    seg_acc /= seg_n
    levenshtein_acc /= levenshtein_n / 100.
    logs = 'CELoss: {:.3f}\t UPLoss: {:.3f}\t SA: {:.3f}%\t LDA: {:.3f}%\t'
    print(logs.format(eval_ce_loss / len(loader), eval_ctc_loss / len(loader), seg_acc, levenshtein_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepPower Evaluate')
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--path', default='results/model_final', type=str, help='load_path')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print('Loading data...')
    data = RaplLoader(batch_size=args.batch_size, num_workers=args.workers)
    dataloader = data.get_loader()

    print('Loading model...')
    net = MateModel_Stru.Model(num_classes=data.num_classes).to(device)
    checkpoint = torch.load(args.path + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    print("Under evaluation...")
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    eval_step(dataloader)
