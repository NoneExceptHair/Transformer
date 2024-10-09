import torch
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu
import nltk



def train(model, train_iter, optimizer, loss_fn, device):
    # global steps
    model.train()
    losses = 0
    total = 0
    hit = 0
    for src, tgt in train_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        tgt_input = tgt[:-1, :]
        logits = model(src, tgt_input)
        tgt_out = tgt[1:, :].reshape(-1)
        output_reshape = logits.reshape(-1, logits.shape[-1])
        # print(tgt_out.shape,tgt_out[0])

        loss = loss_fn(output_reshape, tgt_out)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        total += tgt_out.size(0)
        hit += (output_reshape.max(dim=-1)[1] == tgt_out).sum().item()
        # steps += 1
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lrate(steps)

        losses += loss.item()

    return losses / len(train_iter), round(hit / total * 100, 2)


import numpy as np
import math
import time


def idx_to_word(x, vocab):
    words = []
    for i in x:
        # print(i)
        word = vocab.itos[i]
        if "<" not in word:  # and word not in ['.','\n'] :
            words.append(word)
    # num_token = len(words)
    words = " ".join(words)
    return 0, words


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i : i + n]) for i in range(len(reference) + 1 - n)]
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu2(stats)


def bleu2(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = (
        sum([math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.0
    )
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def evaluate(model, val_iter, loss_fn, device, de_vocab, en_vocab):
    model.eval()
    losses = 0
    batch_bleu = []
    total = 0
    hit = 0
    token_total = 0
    trg_text = []
    pre_text = []

    start_time = time.time()
    for src, tgt in val_iter:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        logits = model(src, tgt_input)
        tgt_out = tgt[1:, :]
        output_reshape = logits.reshape(-1, logits.shape[-1])
        loss = loss_fn(output_reshape, tgt_out.reshape(-1))
        losses += loss.item()
        for j in range(tgt_out.size(1)):
            _, trg_words = idx_to_word(tgt_out[:, j].reshape(-1), de_vocab)
            output_words_index = logits.max(dim=-1)[1][:, j].reshape(-1)
            num_token, output_words = idx_to_word(output_words_index, de_vocab)
            trg_text.append(trg_words)
            pre_text.append(output_words)
            # token_total += num_token
            total += tgt_out.size(0)
            hit += (output_words_index == tgt_out[:,j].reshape(-1)).sum().item()
    end_time = time.time()
    bleu1_scores = []
    bleu2_scores = []
    spend_time = end_time - start_time
    # throughput = token_total / spend_time
    for ref_sentence, hyp_sentence in zip(pre_text, trg_text):
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
    print("speed time:", spend_time)
    # print("throught", throughput)
    print("accuracy:",round(hit / total * 100, 2))

    return losses / len(val_iter), round(hit / total * 100, 2), batch_bleu


# def evaluate(model, val_iter, loss_fn, device,de_vocab,en_vocab):
#     model.eval()
#     losses = 0
#     batch_bleu=[]
#     total = 0
#     hit = 0
#     token_total = 0
#     start_time= time.time()
#     for (src, tgt) in val_iter:
#         src = src.to(device)
#         tgt = tgt.to(device)
#         tgt_input = tgt[:-1, :]
#         logits = model(src, tgt_input)
#         tgt_out = tgt[1:, :]
#         output_reshape = logits.reshape(-1, logits.shape[-1])

#         loss = loss_fn(output_reshape, tgt_out.reshape(-1))
#         losses += loss.item()
#         total_bleu = []

#         for j in range(tgt_out.size(1)):
#         # try:
#             _,trg_words = idx_to_word(tgt_out[:,j].reshape(-1), de_vocab)
#             output_words_index = logits.max(dim=-1)[1][:,j].reshape(-1)
#             num_token,output_words = idx_to_word(output_words_index,de_vocab)
#             # print(trg_words)
#             # print(output_words)

#             bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
#             total_bleu.append(bleu)
#         # # except:

#         # #     pass
#             token_total+=num_token


#         total_bleu = sum(total_bleu) / len(total_bleu)
#         # print(total_bleu)
#         batch_bleu.append(total_bleu)
#         # total += tgt_out.size(0)
#         # hit += (output_reshape.max(dim=-1)[1]==tgt_out).sum().item()
#     end_time = time.time()
#     spend_time = end_time - start_time
#     throughput = token_total / spend_time
#     batch_bleu = sum(batch_bleu) / len(batch_bleu)
#     print("speed time:",spend_time)
#     print("throught", throughput)
#     print("bleu:",batch_bleu)
#     return losses / len(val_iter), round(hit/total*100,2),batch_bleu

def build_vocab(vocab_pth, tokenizer, min_freq=1):
    count = Counter()
    with open(vocab_pth, mode="r", encoding="utf8") as f:
        texts = f.readlines()
        for text in texts:
            count.update(tokenizer(text.lower().rstrip("\n")))

    return Vocab(
        count, min_freq=min_freq, specials=["<unk>", "<pad>", "<bos>", "<eos>"]
    )


def sen2tensor(filepaths, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
    raw_de_iter = iter(open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(open(filepaths[1], encoding="utf8"))
    data = []
    for raw_de, raw_en in zip(raw_de_iter, raw_en_iter):
        de_tensor = torch.tensor(
            [src_vocab[token] for token in src_tokenizer(raw_de.lower().rstrip("\n"))],
            dtype=torch.long,
        )
        en_tensor = torch.tensor(
            [tgt_vocab[token] for token in tgt_tokenizer(raw_en.lower().rstrip("\n"))],
            dtype=torch.long,
        )
        data.append((de_tensor, en_tensor))

    return data


def get_batchtify(PAD_IDX, BOS_IDX, EOS_IDX):
    def batchtify(data_batch):
        de_batch, en_batch = [], []
        for de_item, en_item in data_batch:
            de_batch.append(
                torch.cat(
                    [torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0
                )
            )
            en_batch.append(
                torch.cat(
                    [torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0
                )
            )
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch

    return batchtify


def count_parameters(model):
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()

    return params


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def get_prob(model, ys, memory, device):
    tgt_mask = (
        generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
    ).to(device)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])  # prob: (1,len(en_vocab))
    return prob


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for _ in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (
            generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        ).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break

    return ys


def beam_search(
    model, src, src_mask, max_len, start_symbol, end_symbol, dot, device, beam_k=5
):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask).to(device)
    answers = []

    bos = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)  # (1,len)
    prob = get_prob(model, bos, memory, device)
    next_prob, next_word = torch.max(prob, dim=1)
    answers.append(
        torch.cat([next_prob.view(1, 1, -1), next_word.view(1, 1, -1)], dim=0)
    )
    next_word = next_word.item()
    start = torch.cat(
        [bos, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
    ).to(device)

    # count = 0
    while True:
        # count += 1
        if len(answers) < 2:
            prob = get_prob(model, start, memory, device)  # prob: (1,len(en_vocab))
            next_k_prob, next_k_word = torch.topk(prob, beam_k, dim=1)
            # get k ans
            for p, word in zip(next_k_prob[0], next_k_word[0]):
                temp = torch.cat(
                    [p.view(1, 1, -1), word.view(1, 1, -1)], dim=0
                )  # (2,1,len)
                answers.append(temp)  # 0: prob, 1: seq   (k,2,1,len)
        else:
            for _ in range(beam_k):
                ans = answers.pop(0)  # pop ans (2,1,len)
                seq = ans[1]  # (1,len)
                if seq[0, -1] == end_symbol or seq[0, -1] == dot:
                    answers.append(ans)
                    continue
                # update ys and predict again
                ys = (
                    torch.cat([start.transpose(0, 1), seq], dim=1)
                    .type_as(src.data)
                    .transpose(0, 1)
                    .to(device)
                )
                prob = get_prob(model, ys, memory, device)  # prob: (1,len(en_vocab))
                # get top k ans
                next_k_prob, next_k_word = torch.topk(prob, beam_k, dim=1)
                # gen k new ans
                for p, word in zip(next_k_prob[0], next_k_word[0]):
                    temp = torch.cat(
                        [p.view(1, 1, -1), word.view(1, 1, -1)], dim=0
                    )  # (2,1,1)
                    answers.append(torch.cat([ans, temp], dim=2))  # (2,1,len) + (2,1,1)

        beam_score = torch.tensor(
            [
                torch.sum(torch.tensor([p for p in ans[0, 0]])) / len(ans[0, 0])
                for ans in answers
            ]
        )
        _, top_k_idx = torch.topk(beam_score, beam_k, dim=0)
        answers = [answers[i] for i in top_k_idx]

        if all(
            [
                ans[1, 0, -1] == end_symbol
                or ans[1, 0, -1] == dot
                or len(ans[1, 0]) > max_len
                for ans in answers
            ]
        ):
            break

    beam_score = torch.tensor(
        [
            torch.sum(torch.tensor([p for p in ans[0, 0]])) / len(ans[0, 0])
            for ans in answers
        ]
    )
    best_answer = answers[torch.argmax(beam_score)]  # best answer tokens

    return best_answer[1, 0].type_as(src.data)


def greedy_search(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for _ in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (
            generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        ).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break

    return ys


def translate(
    model, src, src_vocab, tgt_vocab, src_tokenizer, BOS_IDX, EOS_IDX, mode, device
):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = torch.LongTensor(tokens).reshape(num_tokens, 1)
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    if mode == "beam":
        tgt_tokens = beam_search(
            model,
            src,
            src_mask,
            max_len=num_tokens + 5,
            start_symbol=BOS_IDX,
            end_symbol=EOS_IDX,
            device=device,
            dot=tgt_vocab["."],
            beam_k=5,
        ).flatten()
    elif mode == "greedy":
        tgt_tokens = greedy_search(
            model,
            src,
            src_mask,
            max_len=num_tokens + 5,
            start_symbol=BOS_IDX,
            end_symbol=EOS_IDX,
            device=device,
        ).flatten()

    return len(tgt_tokens),(
        " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens])
        .replace("<bos>", "")
        .replace("<eos>", "")
    )
