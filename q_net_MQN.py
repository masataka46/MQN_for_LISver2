# -*- coding: utf-8 -*-
#prototype 02

import copy
import numpy as np
from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F
import memory_unit_link_l6 as MU_l6
import retrieval
import attention
import linearL4_link




class QNet:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**3  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5  # Data size of history. original: 10^6
    hist_size = 1 #original: 4
    time_M =11

    def __init__(self, use_gpu, enable_controller, dim):
        self.use_gpu = use_gpu
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller
        self.dim = dim

        print("Initializing Q-Network...")

        hidden_dim1 = 64
        #hidden_dim1 = 32
        hidden_dim2 = 128
        hidden_dim3 = 10
        hidden_cont = 100
        
        self.model = FunctionSet(
            l4=linearL4_link.LinearL4_link(self.dim*self.hist_size*self.time_M, hidden_cont, wscale=np.sqrt(2)),
            l5=MU_l6.memory_unit_link(self.dim*self.hist_size*self.time_M, hidden_dim3*hidden_cont, wscale=np.sqrt(2)),
            l6=MU_l6.memory_unit_link(self.dim*self.hist_size*self.time_M, hidden_dim3*hidden_cont, wscale=np.sqrt(2)),
            l7=attention.Attention(hidden_cont, hidden_dim3*hidden_cont, hidden_dim3),
            l8=retrieval.Retrieval(hidden_dim3, hidden_dim3*hidden_cont, hidden_cont),
            l9=F.Bilinear(hidden_cont, hidden_cont, hidden_dim2),
            q_value=F.Linear(hidden_dim2, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, hidden_dim2),
                                               dtype=np.float32))
        )
        if self.use_gpu >= 0:
            self.model.to_gpu()

        self.model_target = copy.deepcopy(self.model)

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optimizer.setup(self.model.collect_parameters())

        # History Data :  D=[s(now & 10history), a, r, s_dash, end_episode_flag]
        # modified to MQN
        self.d = [np.zeros((self.data_size, self.hist_size*self.time_M, self.dim), dtype=np.uint8),
                  np.zeros(self.data_size, dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.int8),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.uint8),
                  np.zeros((self.data_size, 1), dtype=np.bool)]

    def forward(self, state, action, reward, state_dash, episode_end):
        num_of_batch = state.shape[0]
        s = Variable(state)
        s_dash = Variable(state_dash)

        q = self.q_func(s)  # Get Q-value

        # Generate Target Signals
        tmp = self.q_func_target(s_dash)  # Q(s',*)
        if self.use_gpu >= 0:
            tmp = list(map(np.max, tmp.data.get()))  # max_a Q(s',a)
        else:
            tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)

        max_q_dash = np.asanyarray(tmp, dtype=np.float32)
        if self.use_gpu >= 0:
            target = np.asanyarray(q.data.get(), dtype=np.float32)
        else:
            # make new array
            target = np.array(q.data, dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = reward[i] + self.gamma * max_q_dash[i]
            else:
                tmp_ = reward[i]

            action_index = self.action_to_index(action[i])
            target[i, action_index] = tmp_

        # TD-error clipping
        if self.use_gpu >= 0:
            target = cuda.to_gpu(target)
        td = Variable(target) - q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = np.zeros((self.replay_size, self.num_of_actions), dtype=np.float32)
        if self.use_gpu >= 0:
            zero_val = cuda.to_gpu(zero_val)
        zero_val = Variable(zero_val)
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, q

    def stock_experience(self, time,
                        state, action, reward, state_dash,
                        episode_end_flag):
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
        else:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
            self.d[3][data_index] = state_dash
        self.d[4][data_index] = episode_end_flag

    def experience_replay(self, time):
        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))
            #modify s_replay for MQN
            s_replay = np.ndarray(shape=(self.replay_size, self.hist_size*self.time_M, self.dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.d[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.d[1][replay_index[i]]
                r_replay[i] = self.d[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.d[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.d[4][replay_index[i]]

            #modify s_dash_replay to 11times for MQN model
            #print 'now state 1'
            s_dash_tmp = s_dash_replay.reshape(len(s_dash_replay), -1).astype(dtype=np.float32)
            #print 'now state 2'
            s_dash_M= np.ndarray(shape=(self.replay_size, self.hist_size*self.time_M, self.dim), dtype=np.float32)
            #print 'now state 3'
            s_dash_M[:,0] = s_dash_tmp
            #print 'now state 4'
            for i in range(self.time_M - 1):
                s_dash_M[:,i + 1] = s_replay[:,i]

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_M)

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()

    def q_func(self, state):
        h4 = F.relu(self.model.l4(state / 255.0))
        h5 = self.model.l5(state / 255.0)
        h6 = self.model.l6(state / 255.0)
        h7 = F.softmax(self.model.l7(h4, h5))
        h8 = self.model.l8(h7, h6)
        h9 = F.relu(self.model.l9(h4, h8))
        q = self.model.q_value(h9)
        return q

    def q_func_target(self, state):
        h4 = F.relu(self.model_target.l4(state / 255.0))
        h5 = self.model_target.l5(state / 255.0)
        h6 = self.model_target.l6(state / 255.0)
        h7 = F.softmax(self.model_target.l7(h4, h5))
        h8 = self.model_target.l8(h7, h6)
        h9 = F.relu(self.model_target.l9(h4, h8))
        q = self.model_target.q_value(h9)
        return q

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        q = self.q_func(s)
        q = q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print(" Random"),
        else:
            if self.use_gpu >= 0:
                index_action = np.argmax(q.get())
            else:
                index_action = np.argmax(q)
            print("#Greedy"),
        return self.index_to_action(index_action), q

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)
