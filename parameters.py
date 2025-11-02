import os

class parameters:

    NUM_AGENTS = 5

    #Training Parameters
    BATCH_SIZE = 64
    NUM_STEPS_TRAIN = 5       # Number of steps to train for
    REPLAY_MEM_SIZE = 512       # Soft maximum capacity of replay memory
    REPLAY_MEM_REMOVE_STEP = 5    # Check replay memory every REPLAY_MEM_REMOVE_STEP training steps and remove samples over REPLAY_MEM_SIZE capacity
    PRIORITY_ALPHA = 0.6            # Controls the randomness vs prioritisation of the prioritised sampling (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    PRIORITY_BETA = 0.4       # Starting value of beta - controls to what degree IS weights influence the gradient updates to correct for the bias introduced by priority sampling (0 - no correction, 1 - full correction)
    PRIORITY_EPSILON = 0.00001      # Small value to be added to updated priorities to ensure no sample has a probability of 0 of being chosen
    NOISE_SCALE = 0.1               # Scaling to apply to Gaussian noise
    NOISE_DECAY = 0.9999            # Decay noise throughout training by scaling by noise_decay**training_step
    GAMMA = 0.99            # Discount rate (gamma) for future rewards
    N_STEP_RETURNS = 2              # Number of future steps to collect experiences for N-step returns
    CANDLE_N_STEP = 5
    UPDATE_AGENT_EP = 10            # Agent gets latest parameters from learner every update_agent_ep episodes
    TAU = 0.001                     # Parameter for soft target network updates
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MIN_EPSILON = 0.05

    #Parameters for TD3 and DDPG
    NOISE_CLIP = 0.5
    EXPLORATION_NOISE = 0.01
    NOISE_STD = 0.1
    #Trading Parameters
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    #TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.0015  # 거래세 0.15%
    #TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_HOLD = 1  # 홀딩
    ACTION_SELL = 2  # 매도

    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_HOLD, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수


class Agent_Parameter:

    def __init__(self, stock_code,balance = 10000, output_path = ' ', 
            value_network_path = ' ', policy_network_path = ' ',lr = 0.001, max_episode = 100):
        self.balance = balance
        self.output_path = output_path
        self.lr = lr
        self.gamma = 0.9
        self.stock_code = stock_code
        self.max_episode = max_episode


        #network path parameter
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        self.store_value_network_path = value_network_path
        self.store_policy_network_path = policy_network_path
        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(
            output_path, 'epoch_summary_{}'.format(
                stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))
    
class Agent_Memory:

    def __init__(self):
        self.memory_value = []
        self.memory_price = []
        self.memory_num_stocks = []
        self.memory_states = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_policy = []
        self.memory_next_price = []
        self.memory_gamma = []

    def reset(self):
        self.memory_value = []
        self.memory_price = []
        self.memory_num_stocks = []
        self.memory_states = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_policy = []
        self.memory_next_price = []
        self.memory_gamma = []