import time
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.optim.lr_scheduler import StepLR, LambdaLR
import matplotlib.pyplot as plt

import numpy as np
from src.utils import *
from src.my_transformer import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=8, help="batch size")
parser.add_argument("--num-enc", type=int, default=6, help="encoder layers numbers")
parser.add_argument("--num-dec", type=int, default=6, help="decoder layers numbers")
parser.add_argument("--emb-dim", type=int, default=512, help="embedding dimension")
parser.add_argument(
    "--ffn-dim", type=int, default=2048, help="feedforward network dimension"
)
parser.add_argument(
    "--head", type=int, default=8, help="head numbers of multihead attention layer"
)
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--epoch", type=int, default=1000, help="training epoch numbers")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument(
    "--fre", type=int, default=2, help="min frequencies of words in vocabulary"
)

args = parser.parse_args()

model_name = "./models/transformer-6-5-2"
BATCH_SIZE = args.batch
NUM_ENCODER_LAYERS = args.num_enc  # no help, 3 is better
NUM_DECODER_LAYERS = args.num_dec  # no help, 3 is better
EMB_SIZE = args.emb_dim
FFN_HID_DIM = args.ffn_dim
NHEAD = args.head  # no help, hard converge
DROPOUT = args.dropout
NUM_EPOCHS = args.epoch
LEARNING_RATE = args.lr
POS_LN = False
# LR_STEP = 30
# warmup_steps = 4000



if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # pth_base = "./.data/multi30k/task1/raw/"
    # train_pths = ("train.en", "train.de")
    # val_pths = ("val.en", "val.de")
    # test_pths = ("test_2016_flickr.en", "test_2016_flickr.de")

    data_len = 256 # 数据长度
    pth_base = '/home/nx/ycy/GraphLLM/data/iwslt/tokenized/'
    train_pths = ("{}_train.en".format(data_len), "{}_train.de".format(data_len))
    val_pths = ("{}_valid.en".format(data_len), "{}_valid.de".format(data_len))
    test_pths = ("{}_test.en".format(data_len), "{}_test.de".format(data_len))

    train_filepaths = [(pth_base + pth) for pth in train_pths]
    val_filepaths = [(pth_base + pth) for pth in val_pths]
    test_filepaths = [(pth_base + pth) for pth in test_pths]

    de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    de_vocab = build_vocab(train_filepaths[1], de_tokenizer, min_freq=args.fre)
    en_vocab = build_vocab(train_filepaths[0], en_tokenizer, min_freq=args.fre)

    train_data = sen2tensor(
        train_filepaths, en_vocab, de_vocab, en_tokenizer, de_tokenizer
    )
    val_data = sen2tensor(val_filepaths, en_vocab, de_vocab, en_tokenizer, de_tokenizer)
    test_data = sen2tensor(
        test_filepaths, en_vocab, de_vocab, en_tokenizer, de_tokenizer
    )
    # 指定使用第二张 GPU
    torch.cuda.set_device(1)

    # 然后将模型和数据移动到指定的 GPU
    # 使用 CUDA 的第 2 张卡
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    DEVICE = device

    print(device)
    print("train size:", len(train_data))
    print("val size:", len(val_data))
    print("test size:", len(test_data))
    print("de vocab size:", len(de_vocab))
    print("en vocab size:", len(en_vocab))

    SRC_VOCAB_SIZE = len(en_vocab)
    TGT_VOCAB_SIZE = len(de_vocab)

    PAD_IDX = en_vocab["<pad>"]
    BOS_IDX = en_vocab["<bos>"]
    EOS_IDX = en_vocab["<eos>"]

    train_iter = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=get_batchtify(PAD_IDX, BOS_IDX, EOS_IDX),
    )
    valid_iter = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=get_batchtify(PAD_IDX, BOS_IDX, EOS_IDX),
    )
    test_iter = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=get_batchtify(PAD_IDX, BOS_IDX, EOS_IDX),
    )

    transformer = MyTf(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                    EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, PAD_IDX,
                    FFN_HID_DIM, DROPOUT, POS_LN, DEVICE)
    # transformer = model = torch.load("./models/transformer-6-5-2-best.pth.tar")
    print(f"The model has {count_parameters(transformer):,} trainable parameters")
    transformer = transformer.to(device)

    # lrate = lambda step_num: EMB_SIZE**-0.5 * np.minimum(step_num**-0.5,step_num*warmup_steps**-1.5)
    # scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9
    )

    train_loss_curve = []
    val_loss_curve = []
    train_acc_list = []
    val_acc_list = []
    min_val_loss = 999
    # steps = 1

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss,train_acc = train(transformer, train_iter, optimizer, loss_fn, device)

        end_time = time.time()
        val_loss, val_acc, bleu = evaluate(
            transformer, test_iter, loss_fn, device, de_vocab, en_vocab
        )
        #     scheduler.step()

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            transformer.eval()
            torch.save(transformer, model_name + "-best.pth.tar")

        if epoch % 50 == 0:
            transformer.eval()
            torch.save(transformer, model_name + "-ckpt-" + str(epoch) + ".pth.tar")

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(
            (
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train accuracy: {train_acc}%, Val loss: {val_loss:.3f}, Valid accuracy: {val_acc}%, Epoch time = {(end_time - start_time):.3f}s"
            )
        )

    with open("./result/train_loss.txt", "w") as file:
        for item in train_loss_curve:
            file.write(item + "\n")
    with open("./result/train_acc.txt", "w") as file:
        for item in train_acc_list:
            file.write(item + "\n")
    with open("./result/val_loss.txt", "w") as file:
        for item in val_loss_curve:
            file.write(item + "\n")
    with open("./result/val_acc.txt", "w") as file:
        for item in val_acc_list:
            file.write(item + "\n")

    print("min val loss:", min_val_loss)
    plt.plot(train_loss_curve)
    plt.plot(val_loss_curve)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(("train loss", "val loss"))
    plt.savefig("./images/" + model_name.split(sep="/")[-1] + ".png")
    transformer.eval()
    torch.save(transformer, model_name + ".pth.tar")
