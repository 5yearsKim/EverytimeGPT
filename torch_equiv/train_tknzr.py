from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast
from tknzrs.utils import make_mecab_corpus

# https://keep-steady.tistory.com/37 참조

vocab_size    = 32000
limit_alphabet= 6000
min_frequency = 4 
special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[CSEP]', '[MSEP]']



def train_tokenizer(corpus_from, save_path='tknzr/test_tknzr' ):
    tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False)

    tokenizer.train(files=corpus_from,
        vocab_size=vocab_size,
        min_frequency=min_frequency,  # 단어의 최소 발생 빈도, 5
        limit_alphabet=limit_alphabet,  # ByteLevelBPETokenizer 학습시엔 주석처리 필요
        special_tokens=special_tokens,
        show_progress=True)

    tokenizer.save_model(save_path)

def replace_unused(vocab_path):
    with open(vocab_path, 'r') as fr:
        vocab = fr.readlines()
    num_unused = 500
    for i in range(500):
        idx = len(vocab) - num_unused + i
        vocab[idx] = f'[unused{i}]\n'
    with open( vocab_path + '_', 'w') as fw:
        fw.writelines(vocab)

def save_to_huggingface(tknzr_path):
    tokenizer_for_load = BertTokenizerFast.from_pretrained(tknzr_path,
                                                        strip_accents=False,  # Must be False if cased model
                                                        lowercase=False)  # 로드

    special_tokens_dict = {'additional_special_tokens': special_tokens[5:]}
    tokenizer_for_load.add_special_tokens(special_tokens_dict)
    tokenizer_for_load.save_pretrained(tknzr_path)

def test_tknzr():
    tokenizer_daily = BertWordPieceTokenizer('./tknzrs/daily_tknzr/vocab.txt', strip_accents=False, lowercase=False)
    tokenizer= BertWordPieceTokenizer('./tknzrs/sentence_klue/vocab.txt', strip_accents=False, lowercase=False)
    file_from = 'data/sample.txt'
    with open(file_from, 'r') as fr:
        for line in fr:
            print(line)
            outputs = tokenizer.encode(line)
            print('klue: ', outputs.tokens)
            outputs_mecab = tokenizer_daily.encode(line)
            print('daily: ', outputs_mecab.tokens)
            print('------------')

if __name__ == '__main__':
    # train_tokenizer(corpus_from=['data/corpus/mecab_corpus.txt'], save_path='tknzrs/daily_tknzr')
    # replace_unused('tknzrs/daily_tknzr/vocab.txt')
    # save_to_huggingface('tknzrs/daily_tknzr')
    test_tknzr()