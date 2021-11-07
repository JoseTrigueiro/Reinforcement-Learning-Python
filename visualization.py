import numpy as np
import matplotlib.pyplot as plt


def compare2Optimal(policy):
    opt_policy_usable_ace = np.concatenate((np.zeros((3, 10)),
                                            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 1]]),
                                            np.ones((6, 10))), axis=0)
    opt_policy_no_usable_ace = np.concatenate((np.zeros((5, 10)),
                                               np.array([[1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                                         [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                                         [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                                         [1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                                         [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]]),), axis=0)

    usable_ace, no_usable_ace = convertPolicy2Array(policy)

    n_sub_optimal = int(np.sum(np.abs(usable_ace - opt_policy_usable_ace)) + np.sum(
        np.abs(no_usable_ace - opt_policy_no_usable_ace)))
    return n_sub_optimal


def convertPolicy2Array(policy, Q=None):
    usable_ace = np.zeros([10, 10])
    values_usable_ace = np.zeros([10, 10])
    values_no_usable_ace = np.zeros([10, 10])
    no_usable_ace = np.zeros([10, 10])
    for state, action in policy.items():
        # Usable Ace
        if state[2]:
            usable_ace[-(state[0] - 11)][state[1] - 1] = action
            if Q:
                values_usable_ace[-(state[0] - 11)][state[1] - 1] = Q[state][action]
        # No Usable Ace
        else:
            no_usable_ace[-(state[0] - 11)][state[1] - 1] = action
            if Q:
                values_no_usable_ace[-(state[0] - 11)][state[1] - 1] = Q[state][action]
    if Q:
        return usable_ace, no_usable_ace, values_usable_ace, values_no_usable_ace

    return usable_ace, no_usable_ace


def showPolicy(Q, policy):
    usable_ace, no_usable_ace, values_usable_ace, values_no_usable_ace = convertPolicy2Array(policy, Q=Q)

    y_tick_labels = [str(i + 12) for i in range(10)]
    x_tick_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    y_tick_labels.reverse()

    fig_policy, axs = plt.subplots(nrows=2, ncols=2)

    # Policy (Usable Ace)
    ax = axs[0, 0]
    ax.imshow(usable_ace)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('Usable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')
    ax.grid(False)

    # Policy (No Usable Ace)
    ax = axs[1, 0]
    ax.imshow(no_usable_ace)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('No Usable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')

    # Value Function (Usable Ace)
    ax = axs[0, 1]
    im = ax.imshow(values_usable_ace)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('Value Function\nUsable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')

    # Value Function (Usable Ace)
    ax = axs[1, 1]
    im = ax.imshow(values_no_usable_ace)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('Value Function\nNo Usable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')

    fig_policy.set_tight_layout(True)
    return fig_policy


def LearningProgess(winrates, n_sub_optimals):
    winrates = np.array(winrates)
    n_sub_optimals = np.array(n_sub_optimals)
    x_iter_win = np.array([i * 100_000 for i in range(len(winrates))])
    x_iter_opt = np.array([i * 10_000 for i in range(len(n_sub_optimals))])

    fig_learn, axs = plt.subplots(nrows=1, ncols=2)
    # plt.style.use('ggplot')
    ax = axs[0]
    ax.plot(x_iter_win, winrates)
    ax.set_title('Win Rate of Policy Over 1000 Games')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate %')

    ax = axs[1]
    ax.plot(x_iter_opt, n_sub_optimals)
    ax.set_title('Difference from Optimal Policy')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Suboptimal Actions')

    fig_learn.set_tight_layout(True)
    return fig_learn


def LearningProgressComparisonQLearning(results, mode, show_every=100_000):
    n_points_win = len(list(results.values())[0]['winrates'])
    x_iter_win = np.array([i * show_every for i in range(n_points_win)])
    n_points_opt = len(list(results.values())[0]['winrates'])
    x_iter_opt = np.array([i * show_every for i in range(n_points_opt)])
    if mode == 'constant_epsilon':
        param_name = 'eps'
    elif mode == 'decay_epsilon':
        param_name = 'eps_decay'
    elif mode == 'boltzmann_exploration':
        param_name = 'temp_decay'

    fig_win, ax_win = plt.subplots()
    ax_win.set_title('Win Rate of Policy Over 10000 Games')
    ax_win.set_xlabel('Iteration')
    ax_win.set_ylabel('Win Rate %')

    fig_opt, ax_opt = plt.subplots()
    ax_opt.set_title('Difference from Optimal Policy')
    ax_opt.set_xlabel('Iteration')
    ax_opt.set_ylabel('Number of Suboptimal Actions')

    for param, step_size in results.keys():
        ax_win.plot(x_iter_win, np.array(results[(param, step_size)]['winrates']),
                    label='{}={:.4f} step_size={:.3f}'.format(param_name, param, step_size))
        ax_opt.plot(x_iter_opt, np.array(results[(param, step_size)]['n_sub_optimals']),
                    label='{}={:.4f} step_size={:.3f}'.format(param_name, param, step_size))

    ax_win.legend()
    ax_opt.legend()

    return fig_opt, fig_win
