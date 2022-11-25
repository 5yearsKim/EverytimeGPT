class AverageMeter:
    """code from TNT"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(y_true, y_prob, threshold=0.5):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > threshold
    y_true = y_true > threshold 
    return (y_true == y_prob).sum().item() / y_true.size(0)