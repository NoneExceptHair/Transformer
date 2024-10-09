import torch
from torchtext.data.utils import get_tokenizer
from src.utils import *
import os
import argparse
import time
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="transformer-6-5-2-ckpt-50", help="model name")
parser.add_argument("--fre", type=int, default=2, help="min frequencies of words in vocabulary")
parser.add_argument("--mode", type=str, default="greedy", help="greedy search or beam search")

args = parser.parse_args()

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
model_pth = "./models/"
if not os.path.exists(model_pth):
    os.mkdir(model_pth)
model_name = args.model
model = torch.load(model_pth + model_name + ".pth.tar")
model.eval()

# pth_base = "./.data/multi30k/task1/raw/"
# train_pths = ('train.en', 'train.de')
# val_pths = ('val.en', 'val.de')
# test_pths = ('test_2016_flickr.en', 'test_2016_flickr.de')


data_len = 256 # 数据长度
pth_base = '/home/nx/ycy/GraphLLM/data/iwslt/tokenized/'
train_pths = ("{}_train.en".format(data_len), "{}_train.de".format(data_len))
val_pths = ("{}_valid.en".format(data_len), "{}_valid.de".format(data_len))
test_pths = ("{}_test.en".format(data_len), "{}_test.de".format(data_len))

train_filepaths = [(pth_base + pth) for pth in train_pths]
test_filepaths = [(pth_base + pth) for pth in test_pths]

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

de_vocab = build_vocab(train_filepaths[1], de_tokenizer, min_freq=args.fre)
en_vocab = build_vocab(train_filepaths[0], en_tokenizer, min_freq=args.fre)

BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

'''load test'''
with open(test_filepaths[0], 'r', encoding='utf8') as f:
    test_data = f.readlines()
for i in range(len(test_data)):
    test_data[i] = test_data[i].rstrip("\n").lower()
    
'''update reference.txt'''
with open(test_filepaths[1], 'r', encoding='utf8') as f:
    reference = f.readlines()

for i in range(len(reference)):
    reference[i] = " ".join(en_tokenizer(reference[i])).lower()

with open("reference.txt",'w+') as f:
    f.writelines(reference)

'''make predictions'''
predictions = []
total_tokens = 0
start_time=time.time()
for data in test_data:
    num_tokens,temp_trans = translate(model, data.lower(), en_vocab, de_vocab, en_tokenizer, BOS_IDX, EOS_IDX, args.mode, device)
    predictions.append(temp_trans+"\n")
    total_tokens+=num_tokens
end_time = time.time()
spend_time = end_time - start_time
throughput = total_tokens / spend_time
print("throught:{}tokens/s".format(throughput))
bleu1_scores = []
bleu2_scores = []
spend_time = end_time - start_time
# throughput = token_total / spend_time
for ref_sentence, hyp_sentence in zip(reference, predictions):
    ref_tokens = nltk.word_tokenize(ref_sentence)
    hyp_tokens = nltk.word_tokenize(hyp_sentence)
    bleu_score = sentence_bleu([ref_tokens], hyp_tokens)
    bleu1_scores.append(bleu_score)

    ref_tokens = [de_vocab[word] for word in ref_sentence.split()]
    hyp_tokens = [de_vocab[word] for word in hyp_sentence.split()]
    bleu_score = sentence_bleu([ref_tokens], hyp_tokens)
    bleu2_scores.append(bleu_score)
average_bleu1 = np.mean(bleu1_scores)
average_bleu2 = np.mean(bleu2_scores)
print("平均 nltk BLEU得分:", average_bleu1 * 100)
print("平均 vocab BLEU得分:", average_bleu2 * 100)

'''update predictions.txt'''
with open("predictions.txt",'w+') as f:
    f.writelines(predictions)

os.system("perl ./src/multi-bleu.perl -lc reference.txt < predictions.txt")
# BLEU = 37.28, 71.3/47.0/32.0/22.4 (BP=0.947, ratio=0.948, hyp_len=12382, ref_len=13058)

'''record predictions'''
with open(model_pth + model_name + ".txt",'w+') as f:    
    f.writelines(predictions)