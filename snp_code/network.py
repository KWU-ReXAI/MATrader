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

    def cirtic_target_predict(self, states, polices):
        """ Predict Q-Values using the target network
        """
        states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        target_q1 = self.target_critic1([states,polices])[0]
        target_q2 = self.target_critic2([states,polices])[0]
        return target_q1, target_q2

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
        critic_states = np.array(critic_states).reshape((len(critic_states), self.window_size, self.inp_dim))
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
        return None, actor_loss, loss

    def actor_train_logloss(self, states):
        """ Actor Training
        """
        with tf.GradientTape() as tape:
            logits = self.actor(states)
            q_values = self.critic1([states, logits])[0]
            actor_loss = -tf.reduce_mean(q_values)
            #loss_function = tf.keras.losses.CategoricalCrossentropy()
            #entropy_loss = loss_function(y_true=imitation_action,y_pred=logits)
            #predPrice = self.price_net(states)
            #price_loss = tf.losses.MSE(y_true=realPrice,y_pred=predPrice)
        grads_actor = tape.gradient(actor_loss,self.actor.trainable_variables)
        grads_actor = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_actor]
        self.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        return actor_loss

    def imitative_train(self, states,imitation_action):
        with tf.GradientTape() as tape:
            logits = self.actor(states)
            loss_function = tf.keras.losses.CategoricalCrossentropy()
            entropy_loss = loss_function(y_true=imitation_action,y_pred=logits)
        grads_action = tape.gradient(entropy_loss,self.actor.trainable_variables)
        grads_action = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_action]
        self.optimizer.apply_gradients(zip(grads_action, self.actor.trainable_variables))
        return entropy_loss

    def price_train(self,states,realPrice):
        states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        with tf.GradientTape() as tape:
            predPrice = self.price_net(states)
            price_loss = tf.losses.MSE(y_true=realPrice,y_pred=predPrice)
        grads_price = tape.gradient(price_loss,self.price_net.trainable_variables)
        grads_price = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads_price]
        self.optimizer.apply_gradients(zip(grads_price, self.price_net.trainable_variables))

    def critic_train(self, states, polies, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        states = np.array(states).reshape((len(states), self.window_size, self.inp_dim))
        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            q_pred1 = self.critic1([states, polies])[0]
            q_pred2 = self.critic2([states, polies])[0]
            loss1 = tf.losses.MSE(critic_target,q_pred1)
            loss2 = tf.losses.MSE(critic_target,q_pred2)
            loss = tf.math.minimum(loss1, loss2)
        grads = tape.gradient(loss, self.critic1.trainable_variables)
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.critic1.trainable_variables))
        grads = tape2.gradient(loss, self.critic2.trainable_variables)
        grads = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.critic2.trainable_variables))
        return loss

    def select_action(self, probs):
        action_probs = np.array(probs)
        pred = action_probs[0]
        # 탐험 결정
        action = np.argmax(pred)
        confidence = pred[action]
        return action, confidence

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path + '.h5')
        self.critic1.save_weights(critic_path + '_1.h5')
        self.critic2.save_weights(critic_path + '_2.h5')
    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path + '.h5')
        self.critic1.load_weights(critic_path + '_1.h5')
        self.critic2.load_weights(critic_path + '_2.h5')

class Imitative_network:
    """ Actor Network for the DDPG Algorithm
    """
    def __init__(self, inp_dim, out_dim, lr, tau,hidden_sell):
        self.hidden_sell = hidden_sell
        self.inp_dim = inp_dim + out_dim + self.hidden_sell
        self.act_dim = out_dim
        self.tau = tau; self.lr = lr
        self.actor = self.actor_network(); self.critic = self.critic_network()
        self.target_actor = self.actor_network(); self.target_critic = self.critic_network()
        self.optimizer = tf.optimizers.Adam(lr,decay=0.9)
    def actor_network(self):
        inp = Input([None,self.inp_dim])
        h = GRU(self.hidden_sell)(inp)
        hidden_layer = Dense(16,activation='relu')(h)
        out = Dense(self.act_dim, activation='softmax')(hidden_layer)
        return Model(inp, [out,h])

    def critic_network(self):
        """ Assemble Critic network to predict q-values
        """
        h = Input([self.hidden_sell])
        policy = Input([self.act_dim])
        con_layer = concatenate([h, policy])
        hidden_layer = Dense(16,activation='relu')(con_layer)
        out = Dense(1, activation='sigmoid')(hidden_layer)
        return Model([h, policy], out)

    def actor_predict(self, state):
        """ Action prediction
        """
        state = np.reshape(state,[state.shape[0],1,self.inp_dim])
        policy,h = self.target_actor(state)
        return policy,h

    def actor_target_predict(self, states):
        """ Action prediction (target network)
        """
        states = np.reshape(states,[states.shape[0],1,self.inp_dim])
        policy,h = self.target_actor(states)
        return policy,h

    def cirtic_target_predict(self, states, polices):
        """ Predict Q-Values using the target network
        """
        target_q = self.target_critic([states,polices])[0]
        return target_q

    def transfer_weights(self):
        self.set_network_weights(self.actor,self.target_actor)
        self.set_network_weights(self.critic,self.target_critic)


    def train(self, episodes,gamma,demonstration_epsilon,T):
        """ Train the critic network on batch of sampled experience
        """
        demonstration_initial_h= np.expand_dims(np.zeros(self.hidden_sell),axis=0) #0 step h
        initial_h = np.expand_dims(np.zeros(self.hidden_sell),axis=0) #0 step h
        curr_h = np.expand_dims(np.zeros(self.hidden_sell),axis=0)
        update_prev_policy = np.array([[0.0,0.0,1.0]])
        demonstration = False
        for index in range(T):
            state,prev_h,prev_policy,policy,r,next_state,expert_action,demonstration = episodes[index]
            if demonstration: prev_h= demonstration_initial_h
            elif index>0: prev_h = initial_h 
            if index == 0: update_prev_policy = prev_policy
            prev_h = np.append(prev_h,update_prev_policy,axis=1); prev_h = np.append(prev_h, state,axis=1)
            prev_h = np.reshape(prev_h,[prev_h.shape[0],1,self.inp_dim])
            if index == 0: _,curr_h = self.actor(prev_h)
            with tf.GradientTape() as tape:   
                h = np.append(curr_h,policy,axis=1); h = np.append(h, next_state,axis=1)
                h = np.reshape(h,[h.shape[0],1,self.inp_dim])
                policy_i,next_h = self.actor_target_predict(h)
                target_q = self.cirtic_target_predict(next_h,policy_i)
                y = r + gamma * target_q[0]
                pred_y = self.critic([curr_h,policy])[0][0]
                critic_loss = (y - pred_y)**2
            critic_prob = abs(y-pred_y)
            _,curr_h = self.actor(h)
            grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                policy,h = self.actor(prev_h)
                actor_update_q = self.critic([h,policy])[0]
                expert_q = self.critic([h,expert_action])[0]
                if expert_q > actor_update_q: expert_loss = tf.losses.MSE(policy_i,expert_q)
                else: expert_loss = 0
                actor_loss = -tf.reduce_mean(actor_update_q)
                loss = actor_loss + expert_loss
            if demonstration: demonstration_initial_h = h
            else: initial_h = h
            update_prev_policy = policy
            grads_actor = tape.gradient(loss,self.actor.trainable_variables)
            self.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            priority_prob = critic_prob + abs(actor_loss)
        if demonstration :priority_prob += demonstration_epsilon
        return priority_prob, critic_loss, loss

    def set_network_weights(self,model,target_model):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = model.get_weights(), target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        target_model.set_weights(target_W)

    def select_action(self, probs):
        action_probs = np.array(probs)
        pred = action_probs[0]
        action = np.argmax(pred)
        confidence = pred[action]
        return action, confidence

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path + '.h5')
        self.critic.save_weights(critic_path + '.h5')
    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path + '.h5')
        self.critic.load_weights(critic_path + '.h5')