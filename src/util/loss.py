import torch
from torch.autograd import Variable
import torch.nn.functional as F


def sequence_mask(sequence_length, max_len=None):
    lengths = sequence_length.detach().cpu().numpy()
    if max_len is None:
        max_len = lengths.max()

    mask = torch.zeros(len(lengths), max_len)
    for i in range(len(lengths)):
        mask[i][:lengths[i]] = 1

    return mask


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: FloatTensor, (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: LongTensor, (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    device = logits.device
    batch_size, max_len, vocab_size = logits.shape
    assert target.dim() == 2 and batch_size == target.shape[0] and max_len == target.shape[1]
    assert length.dim() == 1 and batch_size == length.shape[0]

    logits_flat = logits.reshape(batch_size * max_len, vocab_size)

    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    target_flat = target.reshape(batch_size * max_len, 1)

    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)   # (batch_size * max_len, 1)

    losses = losses_flat.reshape(batch_size, max_len)   # (batch_size, max_len)

    mask = sequence_mask(sequence_length=length, max_len=max_len)   # (batch_size, max_len)
    mask = mask.to(losses.device)
    losses = losses * mask
    loss = losses.sum() / length.float().sum().to(device)
    return loss



