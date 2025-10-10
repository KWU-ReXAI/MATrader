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

# =================================================================
# PPO용 리플레이 버퍼
# =================================================================

class PPOReplayBuffer:
    def __init__(self, num_agents, obs_dim, act_dim):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clear()

    def add(self, obs, action, reward, done, action_log_prob, value_pred):
        """ 에피소드의 한 스텝 데이터를 버퍼에 추가합니다. """
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_log_probs.append(action_log_prob)
        self.value_preds.append(value_pred)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """ 에피소드가 끝나면 호출하여 반환값과 어드밴티지를 계산합니다. """
        gae = 0
        self.returns = [0] * len(self.rewards)
        self.advantages = [0] * len(self.rewards)

        # 마지막 스텝부터 역순으로 계산
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.value_preds[step + 1]
                next_done = self.dones[step + 1]

            delta = self.rewards[step] + gamma * next_value * (1 - next_done) - self.value_preds[step]
            gae = delta + gamma * gae_lambda * (1 - next_done) * gae
            self.returns[step] = gae + self.value_preds[step]
            self.advantages[step] = gae

    def sample(self, num_mini_batch=1):
        """ 학습을 위한 미니배치를 샘플링하는 제너레이터 """
        # 데이터를 numpy 배열로 변환
        obs = np.array(self.obs)
        actions = np.array(self.actions)
        action_log_probs = np.array(self.action_log_probs)
        value_preds = np.array(self.value_preds)
        returns = np.array(self.returns)
        advantages = np.array(self.advantages)

        # 어드밴티지 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = len(self.obs)
        mini_batch_size = batch_size // num_mini_batch

        rand_ids = np.random.permutation(batch_size)

        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            mb_ids = rand_ids[start_idx:end_idx]

            yield (obs[mb_ids], actions[mb_ids], action_log_probs[mb_ids],
                   value_preds[mb_ids], returns[mb_ids], advantages[mb_ids])

    def clear(self):
        """ 버퍼를 초기화합니다. """
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.action_log_probs = []
        self.value_preds = []
        self.returns = []
        self.advantages = []