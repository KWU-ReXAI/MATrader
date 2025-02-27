import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy

# 재현성을 위한 시드 설정
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# ------------------------------
# 1. Actor 네트워크
#  - gate_weight, gate_bias를 이용해 입력 x에 대한 게이트 적용
#  - LSTM으로 상태 인코딩 후, 
#    (1) Policy Head (softmax)
#    (2) Price Head (회귀) 두 가지 출력을 모두 계산
# ------------------------------
class Actor(nn.Module):
    def __init__(self, inp_dim, act_dim, window_size, units):
        super(Actor, self).__init__()
        self.inp_dim = inp_dim
        self.act_dim = act_dim
        self.window_size = window_size
        self.units = units
        
        # Actor 전용 gate 파라미터
        self.gate_weight = nn.Parameter(torch.rand(1, inp_dim))
        self.gate_bias = nn.Parameter(torch.rand(1, inp_dim))
        
        # LSTM layer (batch_first=True로 입력 shape을 (batch, seq, feature)로 가정)
        self.lstm = nn.LSTM(input_size=inp_dim, hidden_size=units, batch_first=True)
        
        # (A) Policy Head: 은닉층 + softmax 출력
        self.actor_head = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, act_dim),
            nn.Softmax(dim=-1)
        )
        
        # (B) Price Head: 은닉층 + 1차원 회귀 출력
        self.price_head = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1)
        )
        
    def forward(self, x):
        # x: (batch, window_size, inp_dim)
        # Gate 계산
        gate = torch.sigmoid(self.gate_weight * x + self.gate_bias)
        weighted = gate * x
        
        # LSTM: 마지막 타임스텝 hidden state 사용
        lstm_out, _ = self.lstm(weighted)  # (batch, window_size, units)
        lstm_out = lstm_out[:, -1, :]      # (batch, units)
        
        # 두 개의 헤드
        policy = self.actor_head(lstm_out)   # (batch, act_dim)
        price = self.price_head(lstm_out)    # (batch, 1)
        
        return policy, price

# ------------------------------
# 2. Critic 네트워크
#  - gate_weight, gate_bias를 이용해 상태(state)에 대한 게이트 적용
#  - LSTM으로 상태 인코딩 후, 액션(action)과 결합해 Q-value 예측
# ------------------------------
class Critic(nn.Module):
    def __init__(self, inp_dim, act_dim, window_size, units):
        super(Critic, self).__init__()
        self.inp_dim = inp_dim
        self.act_dim = act_dim
        self.window_size = window_size
        self.units = units
        
        # Critic 전용 gate 파라미터
        self.gate_weight = nn.Parameter(torch.rand(1, inp_dim))
        self.gate_bias = nn.Parameter(torch.rand(1, inp_dim))
        
        self.lstm = nn.LSTM(input_size=inp_dim, hidden_size=units, batch_first=True)
        self.state_fc = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU()
        )
        self.action_fc = nn.Sequential(
            nn.Linear(act_dim, units),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(units * 2, units),
            nn.ReLU(),
            nn.Linear(units, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state, action):
        # state: (batch, window_size, inp_dim), action: (batch, act_dim)
        gate = torch.sigmoid(self.gate_weight * state + self.gate_bias)
        weighted = gate * state
        
        lstm_out, _ = self.lstm(weighted)   # (batch, window_size, units)
        lstm_out = lstm_out[:, -1, :]       # (batch, units)
        
        s_layer = self.state_fc(lstm_out)
        a_layer = self.action_fc(action)
        concat = torch.cat([s_layer, a_layer], dim=1)
        out = self.fc(concat)  # (batch, 1)
        return out

# ------------------------------
# 3. TD3 Network
#  - Actor & Critic1 & Critic2 (+ Target)
#  - 각각이 고유한 gate 파라미터를 가짐
#  - Soft Update, Optimizer, 학습 로직 등
# ------------------------------
class TD3_network(nn.Module):
    def __init__(self, inp_dim, act_dim, lr, tau, window_size):
        super(TD3_network, self).__init__()
        self.inp_dim = inp_dim
        self.act_dim = act_dim
        self.window_size = window_size
        self.units = 128
        self.tau = tau
        self.lr = lr

        # 디바이스 설정 (GPU 사용 가능 시 cuda 사용)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (1) Actor & Target Actor
        self.actor = Actor(inp_dim, act_dim, window_size, self.units).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)

        # (2) Critic1, Critic2 & Target Critic1, Critic2
        self.critic1 = Critic(inp_dim, act_dim, window_size, self.units).to(self.device)
        self.critic2 = Critic(inp_dim, act_dim, window_size, self.units).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

        # Optimizer 설정
        # - Actor 파라미터 중에서도 actor_head + LSTM + gate 파라미터는 policy 업데이트
        # - price_head + LSTM + gate 파라미터는 price 업데이트
        # - 필요에 따라 원하는 방식으로 나누어 학습할 수 있음
        self.optimizer_a = optim.Adam(
            list(self.actor.actor_head.parameters()) +
            list(self.actor.lstm.parameters()) +
            [self.actor.gate_weight, self.actor.gate_bias],
            lr=lr
        )
        self.optimizer_p = optim.Adam(
            list(self.actor.price_head.parameters()) +
            list(self.actor.lstm.parameters()) +
            [self.actor.gate_weight, self.actor.gate_bias],
            lr=lr
        )
        self.optimizer_c1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.optimizer_c2 = optim.Adam(self.critic2.parameters(), lr=lr)

    # ------------------------------
    # 타깃 네트워크 soft update
    # ------------------------------
    def transfer_weights(self):
        self.soft_update(self.actor, self.target_actor, self.tau)
        self.soft_update(self.critic1, self.target_critic1, self.tau)
        self.soft_update(self.critic2, self.target_critic2, self.tau)
        
    def soft_update(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    # ------------------------------
    # Actor 추론
    # ------------------------------
    def actor_predict(self, state):
        # state는 (window_size, inp_dim) 또는 (batch, window_size, inp_dim)라고 가정
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        policy, _ = self.actor(state)
        return policy.detach().cpu().numpy()

    def actor_target_predict(self, states):
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        if states.dim() == 2:
            states = states.unsqueeze(0)
        policy, _ = self.target_actor(states)
        return policy.detach().cpu().numpy()

    # ------------------------------
    # Critic 타깃 추론
    # ------------------------------
    def critic_target_predict(self, states, policies):
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        if isinstance(policies, np.ndarray):
            policies = torch.tensor(policies, dtype=torch.float32, device=self.device)
        q1 = self.target_critic1(states, policies)
        q2 = self.target_critic2(states, policies)
        return q1.detach().cpu().numpy(), q2.detach().cpu().numpy()

    # ------------------------------
    # Actor 학습 (정책 + 가격)
    # ------------------------------
    def actor_train(self, states, realPrice, critic_states):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        critic_states_tensor = torch.tensor(critic_states, dtype=torch.float32, device=self.device)
        realPrice_tensor = torch.tensor(realPrice, dtype=torch.float32, device=self.device)

        policy, predPrice = self.actor(states_tensor)
        # Critic1을 통해 Q-value 계산
        q_values = self.critic1(critic_states_tensor, policy)
        
        # (1) Actor Loss
        actor_loss = -torch.mean(q_values)
        # (2) Price Loss
        price_loss = F.mse_loss(predPrice, realPrice_tensor)

        # Actor Optimizer
        self.optimizer_a.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.optimizer_a.step()

        # Price Optimizer
        self.optimizer_p.zero_grad()
        price_loss.backward()
        self.optimizer_p.step()

        return None, actor_loss.item(), actor_loss.item()

    # ------------------------------
    # Actor 학습 (로그 손실: imitation 없이 critic으로만)
    # ------------------------------
    def actor_train_logloss(self, states):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        policy, _ = self.actor(states_tensor)
        q_values = self.critic1(states_tensor, policy)
        actor_loss = -torch.mean(q_values)

        self.optimizer_a.zero_grad()
        actor_loss.backward()
        # 기울기 클리핑
        for group in self.optimizer_a.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
        self.optimizer_a.step()
        
        return actor_loss.item()

    # ------------------------------
    # 모방 학습 (Behavior Cloning)
    # ------------------------------
    def imitative_train(self, states, imitation_action):
        # imitation_action: one-hot (batch, act_dim)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        imitation_tensor = torch.tensor(imitation_action, dtype=torch.float32, device=self.device)
        
        policy, _ = self.actor(states_tensor)
        loss = -torch.mean(torch.sum(imitation_tensor * torch.log(policy + 1e-8), dim=1))

        self.optimizer_a.zero_grad()
        loss.backward()
        for group in self.optimizer_a.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
        self.optimizer_a.step()
        
        return loss.item()

    # ------------------------------
    # 가격 예측 학습
    # ------------------------------
    def price_train(self, states, realPrice):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        realPrice_tensor = torch.tensor(realPrice, dtype=torch.float32, device=self.device)
        realPrice_tensor = torch.reshape(realPrice_tensor, (-1,1))

        _, predPrice = self.actor(states_tensor)
        price_loss = F.mse_loss(predPrice, realPrice_tensor)

        self.optimizer_p.zero_grad()
        price_loss.backward()
        for group in self.optimizer_p.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
        self.optimizer_p.step()

    # ------------------------------
    # Critic 학습 (두 Critic의 오차 중 최소값)
    # ------------------------------
    def critic_train(self, states, policies, critic_target):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        policies_tensor = torch.tensor(policies, dtype=torch.float32, device=self.device)
        critic_target_tensor = torch.tensor(critic_target, dtype=torch.float32, device=self.device)

        q_pred1 = self.critic1(states_tensor, policies_tensor)
        q_pred2 = self.critic2(states_tensor, policies_tensor)
        loss1 = F.mse_loss(q_pred1, critic_target_tensor)
        loss2 = F.mse_loss(q_pred2, critic_target_tensor)
        loss = torch.min(loss1, loss2)

        self.optimizer_c1.zero_grad()
        self.optimizer_c2.zero_grad()

        loss.backward()  # backward는 단 한 번
        
        for group in self.optimizer_c1.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
        self.optimizer_c1.step()

        for group in self.optimizer_c2.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
        self.optimizer_c2.step()

        return loss.item()

    # ------------------------------
    # 액션 선택 (argmax)
    # ------------------------------
    def select_action(self, probs):
        # probs: (1, act_dim)
        action_probs = np.array(probs)
        pred = action_probs[0]
        action = np.argmax(pred)
        confidence = pred[action]
        return action, confidence

    # ------------------------------
    # 모델 저장/불러오기
    # ------------------------------
    def save(self, actor_path, critic_path):
        # Actor
        torch.save(self.actor.state_dict(), actor_path + '.pt')
        # Critic
        torch.save({
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }, critic_path + '.pt')

    def load_weights(self, actor_path, critic_path):
        # Actor
        self.actor.load_state_dict(torch.load(actor_path + '.pt', map_location=self.device))
        # Critic
        critic_dict = torch.load(critic_path + '.pt', map_location=self.device)
        self.critic1.load_state_dict(critic_dict['critic1'])
        self.critic2.load_state_dict(critic_dict['critic2'])

    