import mecab
from tqdm import tqdm

mc = mecab.MeCab()

def process_mecab(sent):
    parsed = mc.pos(sent)
    holder = []
    for word, pos in parsed:
        if sent[0] == ' ':
            holder.append(' ')
        start_idx = sent.find(word)
        end_idx = start_idx + len(word)
        # J / E -> ' ##를 '
        if pos.startswith('J') or pos.startswith('E'):
            holder.append(' ')
            word = '##' + word + ' '
        # XS (~이), VCP(~하다)
        elif pos.startswith('XS') or pos.startswith('VCP'):
            holder.append(' ')
            word = '##' + word
        holder.append(word)
        if end_idx < len(sent):
            sent = sent[end_idx:]
    return ''.join(holder) 

def make_mecab_corpus(files_from, file_to):
    with open(file_to, 'w') as fw:
        for file_from in files_from:
            with open(file_from, 'r') as fr:
                for line in tqdm(fr):
                    sent = process_mecab(line)
                    fw.write(sent)
                    fw.write('\n')