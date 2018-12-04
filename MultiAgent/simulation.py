import numpy as np
import matplotlib.pyplot as plt
import math
import environment as en
import agent as ag


def simulation1(simulation_num, episode_num, rewards_type_num):
    labels = ["RS + RS", "RS-OPT", "UCB1T", "e-greedy(0.0025)", "e-greedy_pair(0.005)", "greedy"]
    accuracy = np.zeros((len(labels), episode_num))
    regrets = np.zeros((len(labels), episode_num))
    r_share = np.zeros(episode_num)

    bandit = en.Deterministic_Bandit(rewards_type_num)
    action_num = bandit.bandit_num
    state_num = 2
    end_state = 1
    r = ((rewards_type_num - 1) + (rewards_type_num - 2)) / 20

    agent_list = [ag.RS_multi('ql', state_num, action_num, 2, r),
                  ag.RS('ql', state_num, action_num, r),
                  ag.UCB1T('ql', state_num, action_num),
                  ag.e_Greedy('ql', state_num, action_num, 0.0025),
                  ag.e_Greedy_multi('ql', state_num, action_num, 2, 0.005),
                  ag.Greedy('ql', state_num, action_num)]

    for sim in range(simulation_num):
        print(sim + 1)

        for i, agent in enumerate(agent_list):
            agent.reset_params()
            regret = 0

            for epi in range(episode_num):
                current_state = 0
                # 腕の選択
                current_action = agent.select_arm(current_state)
                # 報酬の観測
                reward = bandit.get_reward(current_state, current_action)
                # 価値の更新
                agent.update(current_state, current_action, reward, end_state, epi+1)
                # accuracy
                accuracy[i][epi] += bandit.get_correct(current_action)
                # regret
                regret += bandit.get_regret(current_action)
                regrets[i][epi] += regret
                # r
                if i == 0:
                    r_share[epi] += agent.get_r()

            # print(regret)

    accuracy /= simulation_num
    regrets /= simulation_num
    r_share /= simulation_num

    plt.xlabel('episode')
    plt.ylabel('accuracy')
    # plt.xscale("log")
    plt.ylim([0.0, 1.1])
    for i, graph in enumerate(accuracy):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("accuracy1")
    plt.show()

    plt.xlabel('episode')
    plt.ylabel('regret')
    # plt.xscale("log")
    for i, graph in enumerate(regrets):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("regret1")
    plt.show()

    plt.xlabel('episode')
    plt.ylabel('r')
    # plt.xscale("log")
    plt.xlim([0.0, 100.0])
    plt.plot(r_share, label="r-share")
    plt.legend(loc="best")
    plt.savefig("r1")
    plt.show()


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

    accuracy /= simulation_num
    regrets /= simulation_num
    r_share /= simulation_num
    reward /= simulation_num

    plt.xlabel('episode')
    plt.ylabel('accuracy')
    # plt.xscale("log")
    plt.ylim([0.0, 1.1])
    for i, graph in enumerate(accuracy):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("1")
    plt.show()

    plt.xlabel('episode')
    plt.ylabel('regret')
    # plt.xscale("log")
    for i, graph in enumerate(regrets):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("1")
    plt.show()

    plt.xlabel('episode')
    plt.ylabel('r')
    # plt.xscale("log")
    for i, graph in enumerate(r_share):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("1")
    plt.show()

    plt.xlabel('episode')
    plt.ylabel('reward')
    # plt.xscale("log")
    for i, graph in enumerate(reward):
        plt.plot(graph, label=labels[i])
    plt.legend(loc="best")
    plt.savefig("1")
    plt.show()


# simulation1(1000, 1000, 10)
simulation2(1000, 5000, 2)
# simulation2(200, 5000, 5)
