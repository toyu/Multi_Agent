import numpy as np
import math


class Environment(object):
    def __init__(self):
        pass

    def get_reward(self, current_state, current_action):
        pass

    def get_correct(self, probability_sum):
        pass

    def get_regret(self, probability_sum):
        pass


class Deterministic_Bandit(Environment):
    def __init__(self, reward_type_num):
        self.bandit = np.asarray([])
        for i in range(reward_type_num):
            for j in range(reward_type_num-i):
                self.bandit = np.append(self.bandit, i/reward_type_num)

        self.bandit_num = len(self.bandit)
        self.max = np.amax(self.bandit)

    def get_reward(self, current_state,  current_action):
        if type(current_action) == type([]):
            rewards = []
            for action in current_action:
                rewards.append(self.bandit[action])
            return rewards
        else:
            return self.bandit[current_action]

    def judge_correct(self, current_action):
        if self.max == self.bandit[current_action]:
            return 1
        else:
            return 0

    def get_correct(self, current_action):
        if type(current_action) == type([]):
            accuracy = 0
            for action in current_action:
                accuracy += self.judge_correct(action)
            return accuracy / len(current_action)
        else:
            return self.judge_correct(current_action)

    def get_regret(self, current_action):
        if type(current_action) == type([]):
            regret = 0
            for action in current_action:
                regret += self.max - self.bandit[action]
            return regret / len(current_action)
        else:
            return self.max - self.bandit[current_action]


class Deterministic_Tree_Bandit(Deterministic_Bandit):
    def __init__(self, layer_num, action_num):
        self.bandit_array = np.array([[0.1, 0.2, 0.3, 0.5],
                                      [0.1, 0.2, 0.8, 0.9],
                                      [0.1, 0.3, 0.4, 0.6],
                                      [0.1, 0.3, 0.6, 0.7],
                                      [0.3, 0.5, 0.6, 0.8],
                                      [0.4, 0.5, 0.6, 0.7]])
        self.layer_num = layer_num
        self.action_num = action_num
        self.set_params()

    def set_params(self):
        self.tree_bandit = []
        self.construction_tree_bandit(self.layer_num)
        self.max = self.get_max(0, 0)

    def construction_tree_bandit(self, layer_num):
        for l in range(layer_num):
            for i in range(int(math.pow(self.action_num, l))):
                bandit_array_length = len(self.bandit_array)
                self.tree_bandit.append(self.bandit_array[np.random.randint(bandit_array_length)])

    def get_max(self, layer, bandit):
        if layer == self.layer_num:
            return 0

        probability_sums = []
        for action in range(self.action_num):
            probability_sum = self.tree_bandit[bandit][action] + self.get_max(layer+1, int(math.pow(self.action_num, layer)+action))
            probability_sums.append(probability_sum)

        return max(probability_sums)

    def get_reward(self, current_states, current_actions):
        rewards = []
        for i in range(len(current_states)):
            rewards.append(self.tree_bandit[current_states[i]][current_actions[i]])
        return rewards

    def judge_correct(self, current_action):
        if self.max == self.tree_bandit[current_action]:
            return 1
        else:
            return 0

    def get_correct(self, reward_sums):
        accuracy = 0
        for reward_sum in reward_sums:
            if self.max == reward_sum:
                accuracy += 1
        return accuracy / len(reward_sums)

    def get_regret(self, reward_sums):
        regret = 0
        for reward_sum in reward_sums:
            regret += self.max - reward_sum
        return regret / len(reward_sums)



