from collections import deque
import random
import numpy as np
import torch
from utilities import transpose_list


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        
        # input_to_buffer = transpose_list(transition)
    
        # for item in input_to_buffer:
        #     self.deque.append(item)
        self.deque.append(transition)

    def sample(self, batchsize):
        """Sample from the buffer
           transpose from: (sample_size, num_items, num_agents, item_size))
           to: (num_items, num_agents, sample_size, item_size)
           Some agents might have different shaped states and actions.
           so we can't just convert to an array and reshape.
           zip(*samples)  - (ss, ni, na, is) -> (ni, ss, na, is)
           then zip(*item) - (ni, ss, na, is) -> (ni, na, ss, is)
        """
        samples = random.sample(self.deque, batchsize)
        samples_t = [list(zip(*item)) for item in zip(*samples)]
        out = []
        for items in samples_t:
            out.append([torch.tensor(np.array(item), dtype=torch.float32) for item in items])
        return out
    

    def __len__(self):
        return len(self.deque)



