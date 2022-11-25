from torch.utils.data import IterableDataset

class GenerationData(IterableDataset):
    def __init__(self, files_from=[]):
        super().__init__()
        self.files_from = files_from

    def __iter__(self):
        for path in self.files_from:
            with open(path, 'r') as fr:
                for row in fr:
                    yield row



