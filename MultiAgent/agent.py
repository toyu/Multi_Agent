import numpy as np
import math


def greedy(values):
    max_values = np.where(values == np.amax(values))
    return np.random.choice(max_values[0])


class Q_learning():
    def __init__(self, learning_rate=0.1, discount_rate=1.0, state_num=None, action_num=None, initial_value=0):
        # self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.state_num = state_num
        self.action_num = action_num
        self.initial_value = initial_value
        if initial_value == 0:
            self.Q = np.zeros((self.state_num, self.action_num))
        elif initial_value == 1:
            self.Q = np.ones((self.state_num, self.action_num))
            self.Q[self.state_num-1].fill(0)

    def reset_params(self):
        if self.initial_value == 0:
            self.Q = np.zeros((self.state_num, self.action_num))
        elif self.initial_value == 1:
            self.Q = np.ones((self.state_num, self.action_num))
            self.Q[self.state_num-1].fill(0)

    def update_Q(self, current_state, current_action, reward, next_state, step):
        TD_error = reward + self.discount_rate * np.amax(self.Q[next_state]) - self.Q[current_state, current_action]
        self.Q[current_state][current_action] += (1 / step) * TD_error

    def get_Q(self):
        return self.Q


class Sarsa(Q_learning):
    def update_Q(self, current_state, current_action, reward, next_state, next_action, step):
        TD_error = (reward
                    + self.discount_rate * np.amax(self.Q[next_state][next_action])
                    - self.Q[current_state][current_action])
        self.Q[current_state][current_action] += (1 / step) * TD_error


class Agent():
    def __init__(self, policy, state_num, action_num, initial_value=0):
        self.state = 0
        self.policy_name = policy
        if policy == 'ql':
            self.policy = Q_learning(state_num=state_num, action_num=action_num, initial_value=initial_value)
        elif policy == 'sarsa':
            self.policy = Sarsa(state_num=state_num, action_num=action_num, initial_value=initial_value)

    def reset_params(self):
        self.policy.reset_params()

    def reset_state(self):
        self.state = 0

    def select_arm(self, current_state):
        pass

    def update(self, current_state, current_action, reward, next_state, step):
        if self.policy_name == 'ql':
            self.policy.update_Q(current_state, current_action, reward, next_state, step)
        elif self.policy_name == 'sarsa':
            self.policy.update_Q(current_state, current_action, reward, next_state, self.select_arm(next_state), step)


class Greedy(Agent):
    def __init__(self, policy, state_num, action_num):
        super().__init__(policy, state_num, action_num, 1)
        self.acton_num = action_num
        self.count_action = np.zeros((state_num, action_num))

    def reset_params(self):
        super().reset_params()
        self.count_action = np.zeros_like(self.count_action)

    def select_arm(self, current_state):
        return greedy(self.policy.get_Q()[current_state])

    def update(self, current_state, current_action, reward, next_state, step):
        self.count_action[current_state][current_action] += 1
        super().update(current_state, current_action, reward, next_state, self.count_action[current_state][current_action])


class e_Greedy(Agent):
    def __init__(self, policy, state_num, action_num, delta):
        super().__init__(policy, state_num, action_num)
        self.e = 1.0
        self.acton_num = action_num
        self.count_action = np.zeros((state_num, action_num))
        self.delta = delta

    def reset_params(self):
        super().reset_params()
        self.count_action = np.zeros_like(self.count_action)
        self.e = 1.0

    def decay_eps(self):
        self.e -= self.delta

    def select_arm(self, current_state):
        if current_state == 0:
            self.decay_eps()

        if self.e > np.random.rand():
            return np.random.randint(self.acton_num)
        else:
            return greedy(self.policy.get_Q()[current_state])

    def update(self, current_state, current_action, reward, next_state, step):
        self.count_action[current_state][current_action] += 1
        super().update(current_state, current_action, reward, next_state, self.count_action[current_state][current_action])


class e_Greedy_multi(Agent):
    def __init__(self, policy, state_num, action_num, agent_num, delta):
        self.agent_num = agent_num
        self.state_num = state_num
        self.action_num = action_num
        self.agents = [(e_Greedy(policy, state_num, action_num, delta)) for i in range(agent_num)]

    def reset_params(self):
        for agent in self.agents:
            agent.reset_params()

    def select_arm(self, current_state):
        actions = []
        for agent in self.agents:
            actions.append(agent.select_arm(current_state))
        return actions

    def update(self, current_state, current_action, rewards, next_state, step):
        max_Qs = []
        max_Q_actions = []
        for i, agent in enumerate(self.agents):
            agent.update(current_state, current_action[i], rewards[i], next_state, step)
            Q = agent.policy.get_Q()
            max_Qs.append(np.amax(Q[current_state]))
            max_Q_actions.append(greedy(Q[current_state]))

        for agent in self.agents:
            for i, max_Q in enumerate(max_Qs):
                agent.policy.Q[current_state][max_Q_actions[i]] = max_Q


class RS(Agent):
    def __init__(self, policy, state_num, action_num, r, gamma=1.0):
        super().__init__(policy, state_num, action_num)
        self.r = np.zeros(state_num)
        self.r[0] = r
        self.state_num = state_num
        self.action_num = action_num
        self.rs = np.zeros((state_num, action_num))
        self.t = np.zeros((state_num, action_num))
        self.t_current = np.zeros((state_num, action_num))
        self.t_post = np.zeros((state_num, action_num))
        self.gamma = gamma

    def reset_params(self):
        super().reset_params()
        self.rs = np.zeros_like(self.rs)
        self.t = np.zeros_like(self.t)
        self.t_current = np.zeros_like(self.t_current)
        self.t_post = np.zeros_like(self.t_post)

    def select_arm(self, current_state):
        Q = self.policy.get_Q()
        self.rs = self.t[current_state] * (Q[current_state] - self.r[current_state])
        return greedy(self.rs)

    def update(self, current_state, current_action, reward, next_state, step):
        Q = self.policy.get_Q()
        self.t_current[current_state][current_action] += 1
        super().update(current_state, current_action, reward, next_state, self.t_current[current_state][current_action])
        action_up = greedy(Q[next_state])
        self.t_post[current_state][current_action] += (1 / step) * (self.gamma * self.t[next_state][action_up] - self.t_post[current_state][current_action])
        self.t[current_state][current_action] = self.t_current[current_state][current_action] + self.t_post[current_state][current_action]


# class RS_multi(Agent):
#     def __init__(self, policy, state_num, action_num, agent_num, gamma=1.0):
#         self.agent_num = agent_num
#         self.state_num = state_num
#         self.action_num = action_num
#         self.agents = [(RS(policy, state_num, action_num, 0, gamma)) for i in range(agent_num)]
#
#     def reset_params(self):
#         for agent in self.agents:
#             agent.reset_params()
#
#     def select_arm(self, current_state):
#         actions = []
#         for agent in self.agents:
#             actions.append(agent.select_arm(current_state))
#         return actions
#
#     def update(self, current_state, current_action, rewards, next_state, step):
#         r = 0
#         max_Qs = []
#         max_Q_actions = []
#         for i, agent in enumerate(self.agents):
#             agent.update(current_state, current_action[i], rewards[i], next_state, step)
#             Q = agent.policy.get_Q()
#             max_Qs.append(np.amax(Q[current_state]))
#             max_Q_actions.append(greedy(Q[current_state]))
#             if max_Qs[i] > r:
#                 r = max_Qs[i]
#
#         for agent in self.agents:
#             agent.r[0] = r
#             for i, max_Q in enumerate(max_Qs):
#                 agent.policy.Q[current_state][max_Q_actions[i]] = max_Q
#
#     def get_r(self):
#         return self.agents[0].r[0]


class RS_multi(Agent):
    def __init__(self, policy, state_num, action_num, agent_num, gamma=1.0):
        self.agent_num = agent_num
        self.state_num = state_num
        self.action_num = action_num
        self.agents = [(RS(policy, state_num, action_num, 0, gamma)) for i in range(agent_num)]

    def reset_params(self):
        for agent in self.agents:
            agent.reset_params()

    def update_r(self,):
        max_Qs_array = np.zeros((self.state_num, self.agent_num))
        max_Q_actions_array = np.zeros((self.state_num, self.agent_num), dtype=np.int32)
        for i, agent in enumerate(self.agents):
            Q = agent.policy.get_Q()
            for j in range(self.state_num):
                max_Qs_array[j][i] = np.amax(Q[j])
                max_Q_actions_array[j][i] = greedy(Q[j])

        max_Qs = []
        max_Q_actions = []
        for i in range(self.state_num):
            max_Qs.append(np.amax(max_Qs_array[i]))
            max_Q_actions.append(max_Q_actions_array[i][greedy(max_Qs_array[i])])

        for agent in self.agents:
            for i in range(self.state_num):
                agent.r[i] = max_Qs[i]

    def select_arm(self, current_state):
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.select_arm(int(current_state[i])))
        return actions

    def update(self, current_states, current_actions, rewards, next_states, step):
        for i, agent in enumerate(self.agents):
            agent.update(current_states[i], current_actions[i], rewards[i], int(next_states[i]), step)

    def get_r(self):
        return self.agents[0].r[0]


class UCB1T(Agent):
    def __init__(self, policy, state_num, action_num):
        super().__init__(policy, state_num, action_num)
        self.count = 0
        self.state_num = state_num
        self.action_num = action_num
        self.t_current = np.zeros((state_num, action_num))
        self.ucb1t = np.zeros((state_num, action_num))
        self.rewards_square = np.zeros((state_num, action_num))

    def reset_params(self):
        super().reset_params()
        self.count = 0
        self.t_current = np.zeros_like(self.t_current)
        self.ucb1t = np.zeros_like(self.ucb1t)
        self.rewards_square = np.zeros_like(self.rewards_square)

    def select_arm(self, current_state):
        if self.count < self.action_num:
            return self.count
        else:
            return greedy(self.ucb1t[current_state])

    def update(self, current_state, current_action, reward, next_state, step):
        self.count += 1
        self.t_current[current_state][current_action] += 1
        self.rewards_square[current_state][current_action] += reward * reward
        super().update(current_state, current_action, reward, next_state, self.t_current[current_state][current_action])
        Q = self.policy.get_Q()

        if self.count >= self.action_num:
            for action in range(self.action_num):
                variance = self.rewards_square[current_state][action] / self.t_current[current_state][action] - Q[current_state][action] * Q[current_state][action]
                v = variance + math.sqrt((2.0 * math.log(self.count)) / self.t_current[current_state][action])
                self.ucb1t[current_state][action] = Q[current_state][action] + math.sqrt((math.log(self.count) / self.t_current[current_state][action]) * min(0.25, v))


class PS(Agent):
    def __init__(self, policy, num_state, action_num, r):
        super().__init__(policy, num_state, action_num)
        self.action_num = action_num
        self.count_action = np.zeros((num_state, action_num))
        self.r = np.asarray([r, 0.0])

    def reset_params(self):
        super().reset_params()
        self.count_action = np.zeros_like(self.count_action)

    def select_arm(self, current_state):
        Q = self.policy.get_Q()
        if np.amax(Q[current_state]) > self.r[current_state]:
            return greedy(Q[current_state])
        else:
            return np.random.randint(self.action_num)

    def update(self, current_state, current_action, reward, next_state, step):
        self.count_action[current_state][current_action] += 1
        super().update(current_state, current_action, reward, next_state, self.count_action[current_state][current_action])
