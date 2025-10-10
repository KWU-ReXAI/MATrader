import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Categorical

# =================================================================
# 유틸리티 함수 및 클래스
# =================================================================

def init(module, weight_init, bias_init, gain=1):
    """ 신경망 모듈 가중치/편향 초기화 """
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    """ 직교 초기화 적용 래퍼 함수 """
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def check(input_tensor):
    """ numpy 배열을 PyTorch 텐서로 변환 """
    return torch.from_numpy(input_tensor) if isinstance(input_tensor, np.ndarray) else input_tensor


def huber_loss(e, d):
    """ Huber 손실 함수 """
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    """ MSE 손실 함수 """
    return e ** 2 / 2


class ValueNorm(nn.Module):
    """ 입력 값 정규화 클래스 """

    def __init__(self, input_shape, device=torch.device("cpu")):
        super(ValueNorm, self).__init__()
        self.device = device
        self.running_mean = torch.zeros(input_shape).to(device)
        self.running_mean_sq = torch.zeros(input_shape).to(device)
        self.debiasing_term = torch.zeros(input_shape).to(device)
        self.epsilon = 1e-5

    def update(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        batch_mean = torch.mean(input_tensor, dim=0)
        batch_sq_mean = torch.mean(input_tensor ** 2, dim=0)

        self.running_mean = self.running_mean * self.debiasing_term + batch_mean * (1 - self.debiasing_term)
        self.running_mean_sq = self.running_mean_sq * self.debiasing_term + batch_sq_mean * (1 - self.debiasing_term)
        self.debiasing_term = self.debiasing_term * 0.99 + (1 - 0.99)

    def normalize(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        normalized_tensor = (input_tensor - self.running_mean) / (
                    torch.sqrt(self.running_mean_sq - self.running_mean ** 2) + self.epsilon)
        return normalized_tensor


# =================================================================
# 행동 생성 함수 (Action Generation)
# =================================================================

def discrete_autoregressive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    """ 순차적 이산 행동 결정 """
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1), **tpdv)
    shifted_action[:, 0, 0] = 1
    output_actions = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log_probs = torch.zeros_like(output_actions, dtype=torch.float32)

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        dist = Categorical(logits=logit)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        action_log_prob = dist.log_prob(action)

        output_actions[:, i, :] = action.unsqueeze(-1)
        output_action_log_probs[:, i, :] = action_log_prob.unsqueeze(-1)

        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_actions, output_action_log_probs


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    """ 병렬적 행동 로그 확률 및 엔트로피 계산 """
    one_hot_action = F.one_hot(action, num_classes=action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1), **tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logits = decoder(shifted_action, obs_rep, obs)
    if available_actions is not None:
        logits[available_actions == 0] = -1e10

    dist = Categorical(logits=logits)
    action_log_probs = dist.log_prob(action).unsqueeze(-1)
    entropy = dist.entropy().unsqueeze(-1)
    return action_log_probs, entropy