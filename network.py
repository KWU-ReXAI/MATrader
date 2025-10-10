import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from utils.util import init_, huber_loss, ValueNorm, check, discrete_parallel_act, discrete_autoregressive_act

# =================================================================
# Multi-Agent Transformer (MAT) 모델 아키텍처
# =================================================================

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))
        if self.masked:
            self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                                 .view(1, 1, n_agent + 1, n_agent + 1))

    def forward(self, key, value, query):
        B, L, D = query.size()
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)

class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                 init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                 init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x

class Encoder(nn.Module):
    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent):
        super(Encoder, self).__init__()
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim), init_(nn.Linear(obs_dim, n_embd), activate=True),
                                         nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, obs):
        obs_embeddings = self.obs_encoder(obs)
        rep = self.blocks(self.ln(obs_embeddings))
        v_loc = self.head(rep)
        return v_loc, rep

class Decoder(nn.Module):
    def __init__(self, action_dim, n_block, n_embd, n_head, n_agent):
        super(Decoder, self).__init__()
        self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                            nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    def forward(self, action, obs_rep, obs):
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        for block in self.blocks:
            x = block(x, obs_rep)
        return self.head(x)

# =================================================================
# MAT 클래스 (학습 및 추론 기능 통합)
# =================================================================

class MultiAgentTransformer(nn.Module):
    def __init__(self, obs_dim, action_dim, n_agent, n_block, n_embd, n_head,
                 lr=5e-4, weight_decay=0, clip_param=0.2,
                 value_loss_coef=1.0, entropy_coef=0.01, huber_delta=10.0,
                 max_grad_norm=10.0, device=torch.device("cpu")):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.max_grad_norm = max_grad_norm

        # 하이퍼파라미터
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.huber_delta = huber_delta

        # 모델 구성
        self.encoder = Encoder(obs_dim, n_block, n_embd, n_head, n_agent)
        self.decoder = Decoder(action_dim, n_block, n_embd, n_head, n_agent)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_normalizer = ValueNorm(1, device=self.device)

        self.to(device)

    def evaluate_actions(self, obs, action, available_actions=None):
        """ 학습을 위해 행동의 로그 확률, 가치, 엔트로피를 계산 """
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv).long()
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = obs.shape[0]
        values, obs_rep = self.encoder(obs)

        action_log_probs, entropy = discrete_parallel_act(
            self.decoder, obs_rep, obs, action, batch_size,
            self.n_agent, self.action_dim, self.tpdv, available_actions
        )

        return (values.squeeze(-1),
                action_log_probs.squeeze(-1),
                entropy.squeeze(-1))

    @torch.no_grad()
    def act(self, obs, available_actions=None, deterministic=False):
        """ 주어진 관측에 대한 행동을 반환 (Inference) """
        self.eval()
        obs = check(obs).to(**self.tpdv).reshape(-1, self.n_agent, obs.shape[-1])
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = obs.shape[0]
        value_preds, obs_rep = self.encoder(obs)

        actions, action_log_probs = discrete_autoregressive_act(
            self.decoder, obs_rep, obs, batch_size,
            self.n_agent, self.action_dim, self.tpdv,
            available_actions, deterministic
        )
        return (actions.squeeze(-1).cpu().numpy(),
                action_log_probs.squeeze(-1).cpu().numpy(),
                value_preds.squeeze(-1).cpu().numpy())

    def _calculate_value_loss(self, values, value_preds_batch, return_batch):
        """ 가치 손실(Critic Loss) 계산 """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        self.value_normalizer.update(return_batch.flatten())
        error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
        error_original = self.value_normalizer.normalize(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped).mean()
        return value_loss

    def update(self, obs_batch, actions_batch, old_action_log_probs_batch,
               value_preds_batch, return_batch, adv_targ, available_actions_batch=None):
        """ PPO 알고리즘으로 모델 파라미터 업데이트 """
        self.train()

        # 텐서 변환 및 디바이스 할당
        obs_batch = check(obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        if available_actions_batch is not None:
            available_actions_batch = check(available_actions_batch).to(**self.tpdv)

        # 현재 정책으로 가치, 로그 확률, 엔트로피 계산
        values, action_log_probs, dist_entropy = self.evaluate_actions(obs_batch, actions_batch,
                                                                       available_actions_batch)

        # 정책 손실 (Actor Loss) 계산
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.min(surr1, surr2).mean()

        # 가치 손실 (Critic Loss) 계산
        value_loss = self._calculate_value_loss(values, value_preds_batch, return_batch)

        # 전체 손실
        loss = policy_loss - dist_entropy.mean() * self.entropy_coef + value_loss * self.value_loss_coef

        # 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": dist_entropy.mean().item()
        }

    def save_model(self, file_path):
        """ 모델 가중치 저장 """
        torch.save(self.state_dict(), file_path+'.pt')
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """ 모델 가중치 불러오기 """
        self.load_state_dict(torch.load(file_path+'.pt', map_location=self.device))
        print(f"Model loaded from {file_path}")
