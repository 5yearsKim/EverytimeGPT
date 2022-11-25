
from dataloader import GenerationData, Collator
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

files_from = ['data/sample.txt']
dset = GenerationData(files_from=files_from)

tknzr = BertTokenizerFast.from_pretrained('tknzrs/nogpt_tknzr')
collator = Collator(tknzr)

loader = DataLoader(dset, batch_size=2, collate_fn=collator)

for item in loader:
    inputs, labels = item
    print(inputs.input_ids)
    print(labels)
    break