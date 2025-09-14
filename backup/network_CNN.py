import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from parameters import parameters

# 재현성을 위한 시드 설정
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ------------------------------
# 1. Actor 네트워크
#  - gate_weight, gate_bias를 이용해 입력 x에 대한 게이트 적용
#  - LSTM으로 상태 인코딩 후, 
#    (1) Policy Head (softmax)
#    (2) Price Head (회귀) 두 가지 출력을 모두 계산
# ------------------------------
class Actor(nn.Module):
	def __init__(self, inp_dim, act_dim, window_size, units, num_stocks):
		super(Actor, self).__init__()
		self.inp_dim = inp_dim
		self.act_dim = act_dim
		self.window_size = window_size
		self.units = units
		
		# Actor 전용 gate 파라미터
		self.gate_weight = nn.Parameter(torch.rand(1, inp_dim))
		self.gate_bias = nn.Parameter(torch.rand(1, inp_dim))

		self.cnn_layer = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2))
			# 필요에 따라 Conv, Pool 레이어 추가
		)

		cnn_output_size = self._get_cnn_output_size(window_size, num_stocks, inp_dim)

		self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=units, batch_first=True)

		
		# (A) Policy Head: 은닉층 + softmax 출력
		self.actor_head = nn.Sequential(
			nn.Linear(units, units),
			nn.ReLU(),
			nn.Linear(units, act_dim),
			nn.Tanh()
		)
		
		# (B) Price Head: 은닉층 + 1차원 회귀 출력
		self.price_head = nn.Sequential(
			nn.Linear(units, units),
			nn.ReLU(),
			nn.Linear(units, act_dim) # 예측해야 하는 가격이 여러개
		)

	def _get_cnn_output_size(self, window_size, num_stocks, inp_dim):
		with torch.no_grad():
			# 배치 크기는 1로 고정하여 더미 입력을 생성
			dummy_input = torch.zeros(1, 1, num_stocks, inp_dim)
			output = self.cnn_layer(dummy_input)
			# output.view(1, -1)는 배치 차원을 제외한 나머지를 모두 flatten
			# .size(1)은 flatten된 피처의 개수를 가져옴
			return output.view(1, -1).size(1)

	def forward(self, x):
		# x: (batch, window_size, inp_dim) 1 10 3 26
		# Gate 계산
		# x의 현재 shape: (batch, window_size, num_stocks, inp_dim)
		batch_size, window, _, _ = x.shape # 1, 10

		# LSTM에 넣기 전, 각 타임스텝의 (num_stocks, inp_dim)을 CNN으로 처리
		# (batch, window, stocks, features) -> (batch * window, stocks, features)
		cnn_in = x.view(batch_size * window, x.size(2), x.size(3)) # 10 3 26
		# (batch * window, stocks, features) -> (batch * window, 1, stocks, features) 채널 차원 추가
		cnn_in = cnn_in.unsqueeze(1) # 10 1 3 26

		# CNN 통과
		cnn_out = self.cnn_layer(cnn_in) # 10 16 1 13

		# CNN 출력을 Flatten
		# (batch * window, C, H, W) -> (batch * window, C*H*W)
		flattened = cnn_out.view(cnn_out.size(0), -1) # 10 208

		# LSTM 입력을 위해 window 차원 복원
		# (batch * window, C*H*W) -> (batch, window, C*H*W)
		lstm_in = flattened.view(batch_size, window, -1) # 1 10 208

		# 이후는 기존과 동일
		lstm_out, _ = self.lstm(lstm_in) # 1 10 128
		lstm_out = lstm_out[:, -1, :] # 1 128

		policy = self.actor_head(lstm_out) * parameters.NUM_ACTIONS
		price = self.price_head(lstm_out)

		return policy, price

# ------------------------------
# 2. Critic 네트워크
#  - gate_weight, gate_bias를 이용해 상태(state)에 대한 게이트 적용
#  - LSTM으로 상태 인코딩 후, 액션(action)과 결합해 Q-value 예측
# ------------------------------
# network.py

class Critic(nn.Module):
	def __init__(self, inp_dim, act_dim, window_size, num_stocks, units):
		super(Critic, self).__init__()
		self.inp_dim = inp_dim
		self.act_dim = act_dim
		self.window_size = window_size
		self.units = units

		# 1. CNN 레이어 정의 (Actor와 동일한 구조 사용)
		self.cnn_layer = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2))
		)

		# 2. CNN 출력 크기 계산
		cnn_output_size = self._get_cnn_output_size(window_size, num_stocks, inp_dim)

		# 3. LSTM 레이어 정의 (입력 크기는 CNN 출력 크기)
		self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=units, batch_first=True)

		# 4. State 처리용 FC 레이어
		self.state_fc = nn.Sequential(
			nn.Linear(units, units),
			nn.ReLU()
		)

		# 5. Action 처리용 FC 레이어
		self.action_fc = nn.Sequential(
			nn.Linear(act_dim, units),
			nn.ReLU()
		)

		# 6. State와 Action을 결합하여 Q-value를 출력하는 최종 FC 레이어
		self.fc = nn.Sequential(
			nn.Linear(units * 2, units),
			nn.ReLU(),
			nn.Linear(units, 1),
			nn.Sigmoid()
		)

	# CNN 출력 크기를 동적으로 계산하기 위한 헬퍼 함수
	def _get_cnn_output_size(self, window_size, num_stocks, inp_dim):
		with torch.no_grad():
			# 배치 크기는 1로 고정하여 더미 입력을 생성
			dummy_input = torch.zeros(1, 1, num_stocks, inp_dim)
			output = self.cnn_layer(dummy_input)
			# output.view(1, -1)는 배치 차원을 제외한 나머지를 모두 flatten
			# .size(1)은 flatten된 피처의 개수를 가져옴
			return output.view(1, -1).size(1)

	def forward(self, state, action):
		# state shape: (batch, window_size, num_stocks, inp_dim) 1 10 3 26
		# action shape: (batch, act_dim)
		batch_size, window, _, _ = state.shape

		# LSTM에 넣기 전, 각 타임스텝의 (num_stocks, inp_dim)을 CNN으로 처리
		# (batch, window, stocks, features) -> (batch * window, stocks, features)
		cnn_in = state.view(batch_size * window, state.size(2), state.size(3))
		# (batch * window, stocks, features) -> (batch * window, 1, stocks, features) 채널 차원 추가
		cnn_in = cnn_in.unsqueeze(1) # 10 3 26

		# CNN 통과
		cnn_out = self.cnn_layer(cnn_in) # 10 16 1 13

		# CNN 출력을 Flatten
		# (batch * window, C, H, W) -> (batch * window, C*H*W)
		flattened = cnn_out.view(cnn_out.size(0), -1) # 10 208

		# LSTM 입력을 위해 window 차원 복원
		# (batch * window, C*H*W) -> (batch, window, C*H*W)
		lstm_in = flattened.view(batch_size, window, -1) # 1 10 208

		# LSTM 통과 후 마지막 타임스텝의 은닉 상태 사용
		lstm_out, _ = self.lstm(lstm_in) # 1 10 128
		lstm_out = lstm_out[:, -1, :] # 1 128

		# State와 Action 결합
		s_layer = self.state_fc(lstm_out)
		a_layer = self.action_fc(action)
		concat = torch.cat([s_layer, a_layer], dim=1)
		out = self.fc(concat)

		return out

# ------------------------------
# 3. TD3 Network
#  - Actor & Critic1 & Critic2 (+ Target)
#  - 각각이 고유한 gate 파라미터를 가짐
#  - Soft Update, Optimizer, 학습 로직 등
# ------------------------------
class TD3_network(nn.Module):
	def __init__(self, inp_dim, act_dim, lr, tau, window_size, num_of_stock):
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
		self.actor = Actor(inp_dim, act_dim, window_size, self.units, num_of_stock).to(self.device)
		self.target_actor = copy.deepcopy(self.actor).to(self.device)

		# (2) Critic1, Critic2 & Target Critic1, Critic2
		self.critic1 = Critic(inp_dim, act_dim, window_size, num_of_stock, self.units).to(self.device)
		self.critic2 = Critic(inp_dim, act_dim, window_size, num_of_stock, self.units).to(self.device)
		self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
		self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)

		# Optimizer 설정
		# - Actor 파라미터 중에서도 actor_head + LSTM + gate 파라미터는 policy 업데이트
		# - price_head + LSTM + gate 파라미터는 price 업데이트
		# - 필요에 따라 원하는 방식으로 나누어 학습할 수 있음
		self.optimizer_a = optim.Adam(self.actor.parameters(),lr=lr)
		self.optimizer_c = optim.Adam(
			list(self.critic1.parameters()) + 
			list(self.critic2.parameters()),
			lr=lr)

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
	# 타깃 네트워크 hard update
	# ------------------------------
	def copy_weights(self):
		self.hard_update(self.actor, self.target_actor, self.tau)
		self.hard_update(self.critic1, self.target_critic1, self.tau)
		self.hard_update(self.critic2, self.target_critic2, self.tau)

	def hard_update(self, source, target, tau):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(
				param.data
			)

	# ------------------------------
	# Actor 추론
	# ------------------------------
	def actor_predict(self, state):
		# state는 (window_size, inp_dim) 또는 (batch, window_size, inp_dim)라고 가정
		if isinstance(state, np.ndarray):
			state = torch.tensor(state, dtype=torch.float32, device=self.device)
		if state.dim() == 3:
			state = state.unsqueeze(0)
		policy, _ = self.actor(state)
		return policy.detach().cpu().numpy()

	def actor_target_predict(self, states):
		if isinstance(states, np.ndarray):
			states = torch.tensor(states, dtype=torch.float32, device=self.device)
		if states.dim() == 3:
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
	def actor_train(self, states, imitation_action, realPrice):
		states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
		imitation_tensor = torch.tensor(imitation_action, dtype=torch.float32, device=self.device)
		realPrice_tensor = torch.tensor(realPrice, dtype=torch.float32, device=self.device)
		# realPrice_tensor = torch.reshape(realPrice_tensor, (-1,1))
		
		policy, predPrice = self.actor(states_tensor)
		q_values = self.critic1(states_tensor, policy)
		actor_loss = -torch.mean(q_values)
		
		imitation_loss = F.cross_entropy(policy, imitation_tensor)

		price_loss = F.mse_loss(predPrice, realPrice_tensor)

		loss = actor_loss + imitation_loss + price_loss

		# Actor Optimizer
		self.optimizer_a.zero_grad()
		loss.backward()
		nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=1.0)
		self.optimizer_a.step()

		return loss.item()

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
		loss = loss1 + loss2

		self.optimizer_c.zero_grad()

		loss.backward()  # backward는 단 한 번
		
		nn.utils.clip_grad_value_(self.critic1.parameters(), clip_value=1.0)
		nn.utils.clip_grad_value_(self.critic2.parameters(), clip_value=1.0)
		self.optimizer_c.step()

		return loss.item()

	# ------------------------------
	# 액션 선택 (argmax)
	# ------------------------------
	# def select_action(self, probs):
	# 	# probs: (1, act_dim)
	# 	action_probs = np.array(probs)
	# 	pred = action_probs[0]
	# 	action = np.argmax(pred)
	# 	confidence = pred[action]
	# 	return action, confidence

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
