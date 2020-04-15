import argparse
import os
import shutil
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config
from config import *
from dataset.datareader import DataReader, Collate
from dataset.vocab import Vocab
from model.deepattn import DeepAttn
from optimizer.optimizer import NoamOpt
from predict import Predictor


def filter(f1_recoder: dict, cur_model: str, args):
    predictor = Predictor(os.path.join(cur_model, 'model.pt'), args.word_vocab, args.label_vocab,
                          'data/dev/word.txt', 'data/dev/label.txt', 'gpu')
    score = predictor.predict()
    f1_recoder[score] = cur_model
    if len(f1_recoder) > 10:
        # remove the worst model
        key = sorted(f1_recoder.keys(), reverse=True)[-1]
        path = f1_recoder.pop(key)
        shutil.rmtree(path)


def plot(points, save_path):
    plt.figure()
    plt.xlabel('step')
    plt.ylabel('loss')
    step = [p[0] for p in points]
    loss = [p[1] for p in points]
    plt.plot(step, loss)
    plt.savefig(save_path)
    plt.close()


def parse_args():
    msg = "train model"
    parser = argparse.ArgumentParser(description=msg)

    msg = "word vocabulary path"
    parser.add_argument("--word_vocab", default='data/train/word_vocab.txt', help=msg)
    msg = "label vocabulary path"
    parser.add_argument("--label_vocab", default='data/train/label_vocab.txt', help=msg)
    msg = "word data path"
    parser.add_argument("--word", default='data/train/word.txt', help=msg)
    msg = "label data path"
    parser.add_argument("--label", default='data/train/label.txt', help=msg)
    msg = "output directory path"
    parser.add_argument("--output", default='result', help=msg)
    msg = "load checkpoint"
    parser.add_argument("--checkpoint", default=None, help=msg)
    msg = 'use gpu'
    parser.add_argument("--gpu", action='store_true', help=msg)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # result save path
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # load vocab
    word_vocab = Vocab(args.word_vocab)
    word_vocab.unk_id = word_vocab.toID(UNK)
    label_vocab = Vocab(args.label_vocab)
    config.WORD_PAD_ID = word_vocab.toID(PAD)
    config.WORD_UNK_ID = word_vocab.toID(UNK)
    config.LABEL_PAD_ID = label_vocab.toID(PAD)
    pred_id = [label_vocab.toID('B-v')]

    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load data
    dataset = DataReader(args.word, args.label, word_vocab, label_vocab)
    dataLoader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=True,
                            collate_fn=Collate(pred_id, WORD_PAD_ID, LABEL_PAD_ID))

    # init model
    model = DeepAttn(word_vocab.size(), label_vocab.size(), feature_dim, model_dim, filter_dim)
    model.to(device)

    if args.checkpoint:
        print('load model: ' + args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint))

    model.train()
    optimiser = NoamOpt(model_dim, 1, warmup_step,
                        torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_epsilon))
    # optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))

    # start train
    all_loss = []
    f1_recoder = {}
    start_time = time.time()
    print('start train')

    for epoch in range(1, epoch + 1):
        epoch_time = time.time()

        # for plot
        loss_record = []
        # save result
        save_path = os.path.join(args.output, str(epoch))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for step, (xs, preds, ys, lengths) in enumerate(dataLoader):
            xs, preds, ys, lengths = xs.to(device), preds.to(device), ys.to(device), lengths.to(device)
            model.zero_grad()
            loss = model(xs, preds, ys)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimiser.step()

            if step % plot_step == 0:
                loss_record.append((step, loss.item()))
                print("epoch: %d, step: %d, loss: %.3f" % (epoch, step, loss.item()))

        print('finished epoch: %d, time: %.2f M' % (epoch, (time.time() - epoch_time) / 60.))

        # save model
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
        # plot
        plot(loss_record, os.path.join(save_path, 'loss.png'))

        filter(f1_recoder, save_path, args)

        if epoch % plot_epoch == 0:
            all_loss.append((epoch, loss_record[-1][1]))
        plot(all_loss, os.path.join(args.output, 'loss.png'))

    plot(all_loss, os.path.join(args.output, 'loss.png'))
    print('finish train: %.2f M' % ((time.time() - start_time) / 60.))
