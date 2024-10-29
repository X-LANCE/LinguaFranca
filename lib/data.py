# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import numpy as np
import random
import torch
from datasets import load_dataset
import hashlib

def hash_string(input_string, length=10):
    # Create a SHA-256 hash object
    sha256 = hashlib.sha256()

    # Update the hash object with the input string encoded as bytes
    sha256.update(input_string.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hex_hash = sha256.hexdigest()

    # Truncate the hash to the desired length
    truncated_hash = hex_hash[:length]

    return truncated_hash


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
# def get_c4(nsamples, seed, seqlen, tokenizer):
#     # Load train and validation datasets
#     traindata = load_dataset('/mnt/lustre/sjtu/home/hcz13/dataset/c4', split='train')
#     valdata = load_dataset('/mnt/lustre/sjtu/home/hcz13/dataset/c4', split='validation')

#     # Generate samples from training set
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if trainenc.input_ids.shape[1] > seqlen:
#                 # with open("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/id.txt", "a") as f:
#                 #     print(f"{i}", file=f, flush=True)
#                 # with open("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/passage.txt", "a") as f:
#                 #     print(traindata[i]['text'], file=f, flush=True)
#                 #     print("-------------------------------------------------", file=f, flush=True)
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))
#     #!!!!这是伪随机！！只选择了大于seqlen的语料！！

#     # with open("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/id.txt", "a") as f:
#     #                 print("----------------------next seed!!!!!!!!!------------------------------", file=f, flush=True)
#     # with open("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/passage.txt", "a") as f:
#     #                 print("----------------------next seed!!!!!!!!!------------------------------", file=f, flush=True)

#     # Prepare validation dataset
#     valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
#     valenc = valenc.input_ids[:, :(256 * seqlen)]
#     valenc = TokenizerWrapper(valenc)


#     return trainloader, valenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('c4', split='train')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for n in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
        input_ids = trainenc.input_ids
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            input_ids = torch.cat([input_ids, trainenc.input_ids], 1)
            if input_ids.shape[1] > seqlen:
                break
        inp = input_ids[:, 0:seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, trainloader


def get_more_bibles(nsamples, seed, seqlen, tokenizer, language):
    # Load train and validation datasets
    # if language!="en":
    #     traindata = load_dataset("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/bible_para.py", lang1="en", lang2=language)
    #     valdata = load_dataset("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/bible_para.py", lang1="en", lang2=language)
    # else:
    #     traindata = load_dataset("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/bible_para.py", lang1="en", lang2="fr")
    #     valdata = load_dataset("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/bible_para.py", lang1="en", lang2="fr")

    traindata = load_dataset("biblenlp-corpus", languages=[language[0:3], "som"], pair="single")
    # valdata = load_dataset("/mnt/lustre/sjtu/home/hcz13/dataset/biblenlp-corpus/biblenlp-corpus.py", languages=[language[0:3]], pair="single")
    try:
        idx = traindata['train'][0]["files"]["file"].index(language)
    except:
        print("Index not found. Use the first instance by default.")
        print(traindata['train'][0])
        idx=0

    print(traindata['train'][0]['translation']['translation'][idx])

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for n in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata['train'][i]['translation']['translation'][idx], return_tensors='pt')
        input_ids = trainenc.input_ids
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata['train'][i]['translation']['translation'][idx], return_tensors='pt')
            input_ids = torch.cat([input_ids, trainenc.input_ids], 1)
            # if len(input_ids[0]) % 128 ==0:
            #     print(f"{len(input_ids[0])} tokens ready")
            if input_ids.shape[1] > seqlen:
                break
        inp = input_ids[:, 0:seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        # print(f"number {n} sample ready.")

    # # Prepare validation dataset
    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)

    return trainloader, trainloader


def get_bibles(nsamples, seed, seqlen, tokenizer, language):
    # Load train and validation datasets
    if language > "en":
        traindata = load_dataset("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/bible_para.py", lang1="en", lang2=language)
    elif language < "en":
        traindata = load_dataset("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/bible_para.py", lang1=language, lang2="en")
    else:
        traindata = load_dataset("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/bible_para.py", lang1="en", lang2="fr")

    print(f"One Calibration Sample: {traindata['train'][0]['translation'][language]}")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for n in range(nsamples):
        i = random.randint(0, len(traindata['train']) - 1)
        trainenc = tokenizer(traindata['train'][i]['translation'][language], return_tensors='pt')
        input_ids = trainenc.input_ids
        while True:
            i = random.randint(0, len(traindata['train']) - 1)
            trainenc = tokenizer(traindata['train'][i]['translation'][language], return_tensors='pt')
            input_ids = torch.cat([input_ids, trainenc.input_ids], 1)
            # if len(input_ids[0]) % 128 ==0:
            #     print(f"{len(input_ids[0])} tokens ready")
            if input_ids.shape[1] > seqlen:
                break
        inp = input_ids[:, 0:seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        # print(f"number {n} sample ready.")
    # # Prepare validation dataset
    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)

    return trainloader, trainloader

def get_cc100(nsamples, seed, seqlen, tokenizer, language):
    # Load train and validation datasets
    traindata = load_dataset('cc100', lang=language)
    print(f"One Calibration Sample: {traindata['train'][0]['text']}")
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for n in range(nsamples):
        i = random.randint(0, len(traindata['train']) - 1)
        trainenc = tokenizer(traindata['train'][i]['text'], return_tensors='pt')
        input_ids = trainenc.input_ids
        while True:
            i = random.randint(0, len(traindata['train']) - 1)
            trainenc = tokenizer(traindata['train'][i]['text'], return_tensors='pt')
            with open("/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/passage_cc100.txt", "a") as f:
                print(traindata['train'][i]['text'], file=f, flush=True)
                print("-------------------------------------------------", file=f, flush=True)
            input_ids = torch.cat([input_ids, trainenc.input_ids], 1)
            if input_ids.shape[1] > seqlen:
                break
        inp = input_ids[:, 0:seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, trainloader

def mix_cc100(nsamples, seed, seqlen, tokenizer, languages):
    if not os.path.exists(f"/mnt/lustre/sjtu/home/hcz13/dataset/save_dataloader/cc100_{hash_string(str(languages+nsamples))}_{seed}.pt"):
        trainloader = []
        for i in range(len(languages)):
            # if languages[i]=="en":
            #     trainloader += get_c4(nsamples[i], seed, seqlen, tokenizer)[0]
            #     ## TODO: 这里混进了c4！！！
            # else:
            trainloader += get_cc100(nsamples[i], seed, seqlen, tokenizer, languages[i])[0]
        torch.save(trainloader, f"/mnt/lustre/sjtu/home/hcz13/dataset/save_dataloader/cc100_{hash_string(str(languages+nsamples))}_{seed}.pt")
        with open("/mnt/lustre/sjtu/home/hcz13/dataset/save_dataloader/hash_table.txt", "a") as f:
            print(f"{str(languages+nsamples)}\t{hash_string(str(languages+nsamples))}", file=f, flush=True)
        assert len(trainloader)==sum(nsamples), f"Length of dataloader : {len(trainloader)}, sample sum: {sum(nsamples)}"
    # ## TODO : 粗暴地解决了存储的问题
    # elif nsamples[0] in [13,8,18,17]:
    #     trainloader = []
    #     for i in range(len(languages)):
    #         # if languages[i]=="en":
    #         #     trainloader += get_c4(nsamples[i], seed, seqlen, tokenizer)[0]
    #         #     ## TODO: 这里混进了c4！！！
    #         # else:
    #         trainloader += get_cc100(nsamples[i], seed, seqlen, tokenizer, languages[i])[0]

    else:
        trainloader = torch.load(f"/mnt/lustre/sjtu/home/hcz13/dataset/save_dataloader/cc100_{hash_string(str(languages+nsamples))}_{seed}.pt")

    return trainloader, trainloader


def get_xlsum(nsamples, seed, seqlen, tokenizer, language):
    # Load train and test datasets
    # traindata = load_dataset('xlsum', language=language)
    testdata = load_dataset('./lib/xlsum.py', language=language, split = "validation[:500]")

    # Encode datasets
    # trainenc = tokenizer(" ".join(traindata['train']['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # # Generate samples from training set
    # random.seed(seed)
    # trainloader = []
    # for _ in range(nsamples):
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))
    return testenc, testenc

def get_xlsum_test(nsamples, seed, seqlen, tokenizer, language):
    # Load train and test datasets
    traindata = load_dataset('/mnt/lustre/sjtu/home/hcz13/remote/wanda_multilingual/lib/xlsum.py', language=language)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for n in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata['train'][i]['text'], return_tensors='pt')
        input_ids = trainenc.input_ids
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata['train'][i]['text'], return_tensors='pt')
            input_ids = torch.cat([input_ids, trainenc.input_ids], 1)
            # if len(input_ids[0]) % 128 ==0:
            #     print(f"{len(input_ids[0])} tokens ready")
            if input_ids.shape[1] > seqlen:
                break
        inp = input_ids[:, 0:seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        # print(f"number {n} sample ready.")
    # # Prepare validation dataset
    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)

    return trainloader, trainloader


def mix_bibles(nsamples, seed, seqlen, tokenizer, languages):
    trainloader = []
    for i in range(len(languages)):
        trainloader += get_bibles(nsamples[i], seed, seqlen, tokenizer, languages[i])[0]
    assert len(trainloader)==sum(nsamples), f"Length of dataloader : {len(trainloader)}, sample sum: {sum(nsamples)}"
    return trainloader, trainloader
    


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, language=None):
    if name=='wikitext2':
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name=="c4":
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if name=="bibles-nlp":
        return get_more_bibles(nsamples, seed, seqlen, tokenizer, language)
    if name=="bibles":
        return get_bibles(nsamples, seed, seqlen, tokenizer, language)
    if name=="cc100":
        return mix_cc100(nsamples, seed, seqlen, tokenizer, language)
    if name=="xlsum":
        return get_xlsum(nsamples, seed, seqlen, tokenizer, language)
    if name=="mix_bibles":
        return mix_bibles(nsamples, seed, seqlen, tokenizer, language)
    
