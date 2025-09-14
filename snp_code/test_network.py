import numpy as np
import tensorflow as tf
import keras.backend as K
import keras
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import Input, Dense,concatenate, LSTM, Lambda, \
                                    Flatten,GaussianNoise,BatchNormalization,GRU,Dropout
import wandb
import random
np.random.seed(42)
random.seed(42)
#tf.random.set_seed(42)
#original: https://github.com/marload/DeepRL-TensorFlow2/blob/master/A3C/A3C_Continuous.py

from keras.layers import Dot, Activation,Concatenate, Reshape
#https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention/attention.py
class TD3_network:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, tau,window_size):
        self.window_size = window_size
        self.inp_dim = inp_dim
        self.act_dim = out_dim
        self.units = 128; self.tau = tau; self.lr = lr
        self.weight = tf.Variable(tf.random.uniform(shape=[1,self.inp_dim]))
        self.bias = tf.Variable(tf.random.uniform(shape=[1,self.inp_dim]))
        self.price_net,self.actor = self.actor_network()
        _,self.target_actor = self.actor_network()
        self.critic1 = self.critic_network(); self.critic2 = self.critic_network()
        self.target_critic1 = self.critic_network(); self.target_critic2 = self.critic_network()
        
        self.optimizer = tf.optimizers.Adam(lr,decay=0.99)

    def actor_network(self):
        inp = Input([self.window_size,self.inp_dim])
        gate = self.weight * inp + self.bias
        gate = tf.math.sigmoid(gate)
        weighted = tf.reshape(gate,[-1,self.inp_dim]) * tf.cast(tf.reshape(inp,[-1,self.inp_dim]),dtype=tf.float32)
        weighted = tf.reshape(weighted,[-1, self.window_size, self.inp_dim])
        gating_layer = LSTM(self.units)(weighted)
        predPrice = self.price_network(gating_layer)
        policy = self.actor_output_network(gating_layer)
        return Model(inp, predPrice), Model(inp, policy)

    def actor_output_network(self, gating_layer):
        hidden_layer = Dense(self.units,activation='relu')(gating_layer)
        out = Dense(self.act_dim, activation='softmax')(hidden_layer)
        return out
    
    def price_network(self,gating_layer):
        hidden_layer = Dense(self.units,activation='relu')(gating_layer)
        out = Dense(1)(hidden_layer)
        return out

    def critic_network(self):
        """ Assemble Critic network to predict q-values
        """
        inp = Input([self.window_size,self.inp_dim])
        gate = self.weight * inp + self.bias
        gate = tf.math.sigmoid(gate)
        weighted = tf.reshape(gate,[-1,self.inp_dim]) * tf.cast(tf.reshape(inp,[-1,self.inp_dim]),dtype=tf.float32)
        weighted = tf.reshape(weighted,[-1, self.inp_dim])
        weighted = tf.reshape(weighted,[-1, self.window_size, self.inp_dim])
        hidden_layer = LSTM(self.units)(weighted)
        policy = Input(shape=[self.act_dim])
        s_layer = Dense(self.units,activation='relu')(hidden_layer)
        a_layer = Dense(self.units,activation='relu')(policy)
        con_layer = concatenate([s_layer, a_layer])
        hidden_layer = Dense(self.units,activation='relu')(con_layer)
        out = Dense(1, activation='sigmoid')(hidden_layer)
        return Model([inp, policy], out)

    def actor_predict(self, state):
        """ Action prediction
        """
        state = np.array(state).reshape((1, self.window_size, self.inp_dim))
        policy = self.actor(state)
        return policy

    def actor_target_predict(self, states):
        """ Action prediction (target network)
        """
        states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        policy = self.target_actor(states)
        return policy

    def cirtic_target_predict(self, states, polies):
        """ Predict Q-Values using the target network
        """
        states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        target_q1 = self.target_critic1([states,polies])[0]
        target_q2 = self.target_critic2([states,polies])[0]
        return target_q1, target_q2

    def attention_vector(self,inputs,hidden_size):
        hidden_states = inputs
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score

        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
        pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    def transfer_weights(self):
        self.set_network_weights(self.actor,self.target_actor)
        self.set_network_weights(self.critic1,self.target_critic1)
        self.set_network_weights(self.critic2,self.target_critic2)

    def set_network_weights(self,model,target_model):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = model.get_weights(), target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        target_model.set_weights(target_W)

    def actor_train(self, states,realPrice,critic_states):
        """ Actor Training
        """
        states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        # Q-Value Gradients under Current Policy
        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            polies = self.actor(states)
            predPrice = self.price_net(states)
            q_values = self.critic1([critic_states, polies])[0]
            
            actor_loss = -tf.reduce_mean(q_values)
            price_loss = tf.losses.MSE(realPrice,predPrice)
            loss = actor_loss
        
        grads_actor = tape.gradient(loss,self.actor.trainable_variables)
        #grads_actor = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_actor]
        grads_price = tape2.gradient(price_loss,self.price_net.trainable_variables)
        #grads_price = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_price]
        self.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.optimizer.apply_gradients(zip(grads_price, self.price_net.trainable_variables))
        return actor_loss

    def actor_train_logloss(self, states, action, advantage,realPrice):
        """ Actor Training
        """
        size = len(states)
        actor_states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            logit,_ = self.actor(states)
            predPrice = self.price_net(actor_states)
            actor_loss = self.compute_loss(size, action, logit, advantage)
            price_loss = tf.losses.MSE(realPrice,predPrice)
            loss = actor_loss + price_loss

        grads_actor = tape.gradient(loss,self.actor.trainable_variables)
        #grads_actor = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_actor]
        grads_price = tape2.gradient(price_loss,self.price_net.trainable_variables)
        #grads_price = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_price]
        self.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.optimizer.apply_gradients(zip(grads_price, self.price_net.trainable_variables))
        return loss

    def critic_train(self, states, polies, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            q_pred1 = self.critic1([states, polies])[0]
            q_pred2 = self.critic2([states, polies])[0]
            loss1 = tf.losses.MSE(critic_target,q_pred1)
            loss2 = tf.losses.MSE(critic_target,q_pred2)
            loss = tf.math.minimum(loss1, loss2)
        grads = tape.gradient(loss, self.critic1.trainable_variables)
        #grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.critic1.trainable_variables))
        grads = tape2.gradient(loss, self.critic2.trainable_variables)
        #grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.critic2.trainable_variables))
        return loss

    def select_action(self, probs):
        action_probs = np.array(probs)
        pred = action_probs[0]
        # 탐험 결정
        action = np.argmax(pred)
        confidence = pred[action]
        return action, confidence

    def compute_loss(self, size, actions, logits, rewards):
        gather_indices = tf.range(size) * tf.shape(logits)[1] + actions
        picked_action_probs = tf.gather(tf.reshape(logits, [-1]), gather_indices)
        losses = -(tf.math.log(picked_action_probs) * rewards)
        loss = tf.reduce_mean(losses)
        return loss

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path + '.h5')
        self.critic1.save_weights(critic_path + '_1.h5')
        self.critic2.save_weights(critic_path + '_2.h5')
    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path + '.h5')
        self.critic1.load_weights(critic_path + '_1.h5')
        self.critic2.load_weights(critic_path + '_2.h5')

class ATTENTION_network:

    def __init__(self, inp_dim, out_dim, lr, window_size):
        self.window_size = window_size
        self.inp_dim = inp_dim
        self.act_dim = out_dim
        self.units = 128; self.lr = lr
        self.prev_action_weight = tf.Variable(tf.random.uniform(shape=[1,self.act_dim]))
        self.attention_weight = tf.Variable(tf.random.uniform(shape=[1,self.inp_dim]))
        self.attention_bias = tf.Variable(tf.random.uniform(shape=[1,self.inp_dim]))
        self.price_net, self.rl_net = self.network()
        self.optimizer = tf.optimizers.Adam(lr,decay=0.99)
    

    def network(self):
        inp = Input([self.window_size, self.inp_dim])
        gate = self.attention_weight * inp + self.attention_bias
        gate = tf.math.sigmoid(gate)
        weighted = tf.reshape(gate,[-1,self.inp_dim]) * tf.cast(tf.reshape(inp,[-1,self.inp_dim]),dtype=tf.float32)
        weighted = tf.reshape(weighted,[-1, self.window_size, self.inp_dim])
        gru = GRU(self.units,return_sequences=True)(weighted)
        envVec = self.attention_vector(gru,self.units)
        predPrice = self.price_network(envVec)
        policy = self.actor_network(envVec)
        return Model(inp, predPrice), Model(inp,policy)
    
    def actor_network(self, attention_vec):
        hidden_layer = Dense(self.units,activation='relu')(attention_vec)
        hidden_layer = Dense(self.units/2,activation='relu')(hidden_layer)
        Dropout_layer = Dropout(0.3)(hidden_layer)
        policy = Dense(self.act_dim, activation='softmax')(Dropout_layer)
        return policy
    def price_network(self,attention_vec):
        hidden_layer = Dense(self.units, activation='relu')(attention_vec)
        hidden_layer = Dense(self.units/2,activation='relu')(hidden_layer)
        Dropout_layer = Dropout(0.3)(hidden_layer)
        out = Dense(1)(Dropout_layer)
        return out

    def predict(self, state):
        sample = np.array(state).reshape((1, self.window_size, self.inp_dim))
        policy = self.rl_net(sample)
        return policy

    def attention_vector(self,inputs,hidden_size):
        hidden_states = inputs
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score

        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
        pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    def compute_loss(self, size, actions, logits, rewards):
        logits += 1e-8
        gather_indices = tf.range(size) * tf.shape(logits)[1] + actions
        picked_action_probs = tf.gather(tf.reshape(logits, [-1]), gather_indices)
        losses = (tf.math.log(picked_action_probs) * rewards)
        return losses

    def train(self, states,realPrice,next_realPrice):
        states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        # Q-Value Gradients under Current Policy
        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            policies = self.rl_net(states)
            predPrice = self.price_net(states)
            U,actions = self.utility(realPrice, next_realPrice, policies)
            actor_loss = self.compute_loss(len(actions),actions,policies,U)
            price_loss = tf.losses.MSE(realPrice,predPrice)
            loss = -tf.reduce_mean(actor_loss) +price_loss 
        grads_rl = tape.gradient(loss,self.rl_net.trainable_variables)
        #grads_actor = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_actor]
        grads_price = tape2.gradient(price_loss,self.price_net.trainable_variables)
        #grads_price = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_price]
        self.optimizer.apply_gradients(zip(grads_rl, self.rl_net.trainable_variables))
        self.optimizer.apply_gradients(zip(grads_price, self.price_net.trainable_variables))
        return loss

    def utility(self, realPrice,next_realPrice, policies, c=0.001):
        actions = tf.argmax(policies, axis=1) # 32
        
        actions = np.array(actions)
        z = np.zeros(len(realPrice))  # PRICE t+1 - PRICE t
        for i in range(len(realPrice)):
            z[i] = next_realPrice[i] - realPrice[i]
        R = np.zeros(len(realPrice))
        p_a = 0
        for i in range(len(actions)):
            if actions[i] == 0 : a = -1
            elif actions[i] == 1 : a = 1
            else : a = 0
            if i == 0: 
                R[i] = a * z[i]; p_a = a
            else: 
                R[i] = a* z[i] - c*abs(a - p_a); p_a = a
        U = sum(R)
        return U,actions

    def select_action(self, probs,epsilon):
        action_probs = np.array(probs)
        pred = action_probs[0]
        if np.random.rand() < epsilon:
            action = np.random.randint(0,self.act_dim-1)
        else : action = np.argmax(pred)
        confidence = pred[action] 
        return action, confidence

    def save_weights(self, net_path):
        self.rl_net.save_weights(net_path + '.h5')

    def load_weights(self, net_path):
        self.rl_net.load_weights(net_path + '.h5')
    