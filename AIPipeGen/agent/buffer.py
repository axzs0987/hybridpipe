import random
import numpy as np
import os

class ReplayBuffer(object):
    def __init__(self, capacity, process_id=None):
        self.capacity = capacity
        self.process_id = process_id
        # if process_id is not None:
        #     if os.path.exists('buffer'+str(self.process_id)+'.npy'):
        #         self.buffer = list(np.load('buffer'+str(self.process_id)+'.npy', allow_pickle=True))
        #     else:
        #         self.buffer = []
        # else:
        self.buffer = []

        # if process_id is not None:
        #     # if os.path.exists('lp_buffer'+str(self.process_id)+'.npy'):
        #     #     self.lp_buffer = list(np.load('lp_buffer'+str(self.process_id)+'.npy', allow_pickle=True))
        #     # else:
        #         self.lp_buffer = []
        # else:
        self.lp_buffer = []

    def add(self, s0, a, r, s1, done, index, fixline_id):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done, index, fixline_id))
        # np.save('buffer'+str(self.process_id)+'.npy', self.buffer)
        # if os.path.exists('/home/chensibei/action_buffer.npy'):
        #     global_buffer = list(np.load('/home/chensibei/action_buffer.npy'))
        # else:
        #     global_buffer = []
        # global_buffer.append((s0[None, :], a, r, s1[None, :], done, index))
        # np.save('/home/chensibei/action_buffer.npy', global_buffer)
        # del global_buffer

    def sample(self, batch_size):
        s0, a, r, s1, done, index, fixline_id = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s0), a, r, np.concatenate(s1), done, index, fixline_id

    def lp_add(self, s0, a, r):
        if len(self.lp_buffer) >= self.capacity:
            self.lp_buffer.pop(0)
        self.lp_buffer.append((s0[None, :], a, r))
        # np.save('lp_buffer'+str(self.process_id)+'.npy', self.lp_buffer)

        # if os.path.exists('/home/chensibei/lp_buffer.npy'):
        #     global_buffer = list(np.load('/home/chensibei/lp_buffer.npy'))
        # else:
        #     global_buffer = []
        # global_buffer.append((s0[None, :], a, r))
        # np.save('/home/chensibei/lp_buffer.npy', global_buffer)
        # del global_buffer

    def lp_sample(self, batch_size):
        s0, a, r = zip(*random.sample(self.lp_buffer, batch_size))
        return np.concatenate(s0), a, r

    def size(self):
        return len(self.buffer)
