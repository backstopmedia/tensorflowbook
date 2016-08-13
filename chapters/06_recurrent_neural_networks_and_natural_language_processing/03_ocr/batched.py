import numpy as np


def batched(data, target, batch_size):
    """
    
    """
    epoch = 0
    offset = 0
    while True:
        old_offset = offset
        offset = (offset + batch_size) % (target.shape[0] - batch_size)

        if offset < old_offset:
            # New epoch, need to shuffle data
            p = np.random.permutation(len(data))
            data = data[p]
            target = target[p]
            epoch += 1

        batch_data = data[offset:(offset + batch_size), :]
        batch_target = target[offset:(offset + batch_size), :]
        yield batch_data, batch_target, epoch
        