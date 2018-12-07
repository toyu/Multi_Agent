import numpy as np
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed
from time import time
import environment as en
import agent as ag


def simulation2(simulation_num, episode_num, layer_num):
    labels = ["agent_num = 1", "agent_num = 2", "agent_num = 3", "agent_num = 4", "agent_num = 5"]
    # labels = ["agent_num = 3"]
    agent_num = (len(labels))
    accuracy = np.zeros((agent_num, episode_num))
    regrets = np.zeros((agent_num, episode_num))
    reward = np.zeros((agent_num, episode_num))
    r_share = np.zeros((agent_num, episode_num))
    action_num = 4
    state_num = sum([int(math.pow(action_num, i)) for i in range(layer_num)]) + 1
    end_state = state_num - 1
    tree_bandit = en.Deterministic_Tree_Bandit(layer_num, action_num)

    agent_list = [ag.RS_multi('ql', state_num, action_num, 1),
                  ag.RS_multi('ql', state_num, action_num, 2),
                  ag.RS_multi('ql', state_num, action_num, 3),
                  ag.RS_multi('ql', state_num, action_num, 4),
                  ag.RS_multi('ql', state_num, action_num, 5)]
    # agent_list = [ag.RS_multi('ql', state_num, action_num, 3)]

    for sim in range(simulation_num):
        print(sim + 1)
        tree_bandit.construction_tree_bandit(layer_num)

        for i, agent in enumerate(agent_list):
            agent.reset_params()
            regret = 0

            for epi in range(episode_num):
                reward_sums = np.zeros(agent.agent_num)
                step = 1
                current_states = np.zeros(agent.agent_num, dtype=np.int32)

                while current_states[0] < end_state:
                    # 腕の選択
                    current_actions = agent.select_arm(current_states)
                    # 報酬の観測
                    rewards = tree_bandit.get_reward(current_states, current_actions)
                    reward_sums += np.array(rewards)
                    # 次状態の観測
                    next_states = np.array([], dtype=np.int32)
                    for i in range(agent.agent_num):
                        next_states = np.append(next_states, current_states[i] * action_num + current_actions[i] + 1)
                    if next_states[0] > (state_num - 2):
                        next_states.fill(end_state)
                    # 価値の更新
                    agent.update(current_states, current_actions, rewards, next_states, step)
                    # 状態の更新
                    current_states = next_states
                    step += 1

                # update r
                agent.update_r()
                # accuracy
                accuracy[i][epi] += tree_bandit.get_correct(reward_sums)
                # regret
                regret += tree_bandit.get_regret(reward_sums)
                regrets[i][epi] += regret
                # reward
                reward[i][epi] += np.sum(reward_sums) / agent.agent_num
                # r_share
                r_share[i][epi] += agent.get_r()

    return np.array([accuracy, regrets, reward, r_share])


def plot_graph(data, agent_num, data_type_num, episode_num, job_num):
    for i in range(data_type_num):
        graphs = np.zeros((agent_num, episode_num))

        for j in range(job_num):
            graphs += data[j][i]

        graphs /= simulation_num
        plt.xlabel('episode')
        plt.ylabel(graph_titles[i])
        # plt.xscale("log")
        for g, graph in enumerate(graphs):
           plt.plot(graph, label=labels[g])
        plt.legend(loc="best")
        plt.savefig(graph_titles[i])
        plt.show()


simulation_num = 10
job_num = 10
simulation_num_per_job = int(simulation_num / job_num)
episode_num = 5000
agent_num = 5
data_type_num = 4
layer_num = 2
labels = ["agent_num = 1", "agent_num = 2", "agent_num = 3", "agent_num = 4", "agent_num = 5"]
graph_titles = ["accuracy", "regret", "reward", "r"]

start = time()

data = Parallel(n_jobs=job_num)([delayed(simulation2)(simulation_num_per_job, episode_num, layer_num) for i in range(job_num)])
plot_graph(data, agent_num, data_type_num, episode_num, job_num)

print('{}秒かかりました'.format(time() - start))

