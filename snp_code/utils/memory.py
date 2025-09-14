import random
import numpy as np

from collections import deque

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size):
        """ Initialization
        """
        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state, critic_state, action, reward, done, new_state, critic_next_state, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """

        experience = (state, critic_state, action, reward, done, new_state, critic_next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []
        # Sample using prorities
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        c_s_batch = np.array([i[1] for i in batch])
        a_batch = np.array([i[2] for i in batch])
        r_batch = np.array([i[3] for i in batch])
        d_batch = np.array([i[4] for i in batch])
        new_s_batch = np.array([i[5] for i in batch])
        c_new_s_batch = np.array([i[6] for i in batch])
        return s_batch, c_s_batch, a_batch, r_batch, d_batch, new_s_batch, c_new_s_batch, idx

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        self.buffer = deque()
        self.count = 0

class Transaction(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size):
        """ Initialization
        """
        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """
        experience = (state, action, reward, done, new_state)
        self.buffer.append(experience)
        self.count += 1
    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []
        # Sample using prorities
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)
        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        return s_batch, a_batch, r_batch, d_batch, new_s_batch, idx

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        self.buffer = deque()
        self.count = 0

class TD3_MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size):
        """ Initialization
        """
        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state, action, critic_state, reward, done, next_state, critic_next_state,gamma, price, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """
        experience = (critic_state, state, action, reward, done, next_state, critic_next_state, gamma, price)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []
        # Sample using prorities
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[1] for i in batch])
        c_s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[2] for i in batch])
        r_batch = np.array([i[3] for i in batch])
        d_batch = np.array([i[4] for i in batch])
        next_s_batch = np.array([i[5] for i in batch])
        c_next_s_batch = np.array([i[6] for i in batch])
        g_batch = np.array([i[7] for i in batch])
        p_batch = np.array([i[8] for i in batch])
        return s_batch, a_batch, c_s_batch, r_batch, d_batch, next_s_batch, c_next_s_batch, g_batch, p_batch, idx

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        self.buffer = deque()
        self.count = 0

class ATTENTION_MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size):
        """ Initialization
        """
        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state,reward, done, prev_price,price, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """
        experience = (state,reward, done, prev_price,price)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []
        # Sample using prorities
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)
        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        r_batch = np.array([i[1] for i in batch])
        d_batch = np.array([i[2] for i in batch])
        pp_batch = np.array([i[3] for i in batch])
        p_batch = np.array([i[4] for i in batch])
        return s_batch, r_batch, d_batch, pp_batch,p_batch

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        self.buffer = deque()
        self.count = 0

class IMI_Transaction(object):
    def __init__(self, buffer_size):
        """ Initialization
        """
        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, episode, done, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """
        experience = (episode, done)
        self.buffer.append(experience)
        self.count += 1
    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []
        # Sample using prorities
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)
        # Return a batch of experience
        episode_batch = np.array([i[0] for i in batch])
        d_batch = np.array([i[1] for i in batch])
        return episode_batch, d_batch, idx

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        self.buffer = deque()
        self.count = 0