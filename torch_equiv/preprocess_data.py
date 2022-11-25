from tknzrs.utils import process_mecab
from tqdm import tqdm
import random


def preprocess_for_lm(sent):
    sent = sent.strip()
    sent = sent.replace('#&#', '[CSEP]').replace('#|#', '[MSEP]')
    holder = []
    def split_and_push(sent):
        if len(sent) < 200:
            holder.append(sent)
        else:
            sent_list = sent.split('[CSEP]')
            if len(sent_list) <= 1:
                idx = len(sent) // 2
                sent1, sent2 = sent[:idx], sent[idx:]
            else:
                idx = len(sent_list) // 2
                sent1, sent2 = '[CSEP]'.join(sent_list[:idx]), '[CSEP]'.join(sent_list[idx:])
            split_and_push(sent1)
            split_and_push(sent2)
    split_and_push(sent)
    return holder

def preprocess_for_corpus(sent):
    sent = sent.strip()
    sent = sent.replace('#&#', ' ').replace('#|#', ' ')
    sent = process_mecab(sent)
    return sent

def main_for_lm():
    files_from = [
        'data/original/everytime.txt',
        'data/original/aihub_messenger.txt',
        'data/original/aihub_sns.txt',
    ]
    holder = []
    for file in files_from:
        print('reading - ', file)
        with open(file, 'r') as fr:
            for i, line in enumerate(fr):
                sent_list = preprocess_for_lm(line)
                holder.extend(sent_list)
    val_len = 10000
    print('shuffle start..')
    random.shuffle(holder)
    print('shuffle done.. writing..')
    with open('data/for_lm/val_lm.txt', 'w') as fw:
        for line in holder[:val_len]:
            fw.write(line)
            fw.write('\n')
    with open('data/for_lm/train_lm.txt', 'w') as fw:
        for line in holder[val_len:]:
            fw.write(line)
            fw.write('\n')

    

def main_for_corpus():
    files_from = [
        # 'data/sample.txt'
        'data/original/news.txt',
        'data/original/everytime.txt',
        'data/original/aihub_sns.txt',
        'data/original/aihub_messenger.txt',
    ]
    with open('data/corpus/mecab_corpus.txt', 'w') as fw:
        for file in files_from:
            print(file)
            with open(file, 'r') as fr:
                for i, line in enumerate(tqdm(fr)):
                    sent= preprocess_for_corpus(line)
                    fw.write(sent)
                    fw.write('\n')

if __name__ == '__main__':
    # main_for_corpus()
    main_for_lm()
