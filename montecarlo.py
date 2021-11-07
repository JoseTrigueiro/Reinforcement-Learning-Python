import blackjack
from itertools import product
import random
import visualization
import numpy as np
import matplotlib.pyplot as plt
import time


class MCLearningAgent:
    def __init__(self, explore_policy='constant epsilon', eps=0.05, natural=False,
                 eps_decay=0.9999, show_every=1000000, evaluate_iter=1000, temp_decay=0.9999, init_temp=20):
        self.Q = self.initializeQ()
        self.winrates = []
        self.natural = natural
        self.show_every = show_every
        self.evaluate_iter = evaluate_iter
        self.env = blackjack.BlackjackEnv(natural=self.natural)
        self.n_sub_optimals = []
        self.min_eps = 0.001
        self.min_temp = 0.1
        if explore_policy == 'constant_epsilon':
            self.explore_policy = self.e_greedy
            self.eps = eps
            self.eps_decay = 1
            self.temp = init_temp
            self.temp_decay = 1
        elif explore_policy == 'decay_epsilon':
            self.explore_policy = self.e_greedy
            self.eps = 1
            self.eps_decay = eps_decay
            self.temp = init_temp
            self.temp_decay = 1
        elif explore_policy == 'boltzmann_exploration':
            self.explore_policy = self.boltzmann_exploration
            self.eps = 1
            self.eps_decay = 1
            self.temp = init_temp
            self.temp_decay = temp_decay

    def e_greedy(self, state_values):
        best_action = state_values.index(max(state_values))
        roll = random.random()
        if roll < self.eps:
            action = random.randint(0, 1)
        else:
            action = best_action
        return action

    def boltzmann_exploration(self, state_values):
        state_values = np.array(state_values)
        exp_values = np.exp(state_values) / self.temp
        probs = exp_values / np.sum(exp_values)
        roll = random.random()
        if roll < probs[0]:
            action = 0
        else:
            action = 1
        return action

    def initializeQ(self):
        return {state: [0, 0] for state in
                product(range(12, 22), range(1, 11), [True, False])}

    def exploringStarts(self, numIterations=10_000):
        self.Q = self.initializeQ()
        numberOfVisits = {state: 0 for state in
                          product(range(12, 22), range(1, 11), [True, False], [0, 1])}

        for i in range(numIterations):
            state = self.env.reset()
            memory = []
            gameEnd = False
            isFirstState = True

            while not gameEnd:
                if state[0] < 12:
                    newState, reward, gameEnd, _ = self.env.step(1)
                elif isFirstState:
                    isFirstState = False
                    action = random.randint(0, 1)
                    newState, reward, gameEnd, _ = self.env.step(action)
                    memory.append((state, action, reward))
                else:
                    state_values = self.Q[state]
                    action = state_values.index(max(state_values))
                    newState, reward, gameEnd, _ = self.env.step(action)
                    memory.append((state, action, reward))
                state = newState

                #     state_values = self.Q[state]
                #     action = self.explore_policy(state_values)
                #     newState, reward, gameEnd, _ = self.env.step(action)
                #     memory.append((state, action, reward))
                # state = newState

            G = 0
            gamma = 1
            for state, action, reward in reversed(memory):
                G = gamma * G + reward
                numberOfVisits[state + (action,)] += 1
                self.Q[state][action] = self.Q[state][action] + G / numberOfVisits[state + (action,)]

            # decay eps
            if self.eps_decay != 1 and self.eps > self.min_eps:
                self.eps *= self.eps_decay
                self.eps = max(self.eps, self.min_eps)

            # temp decay
            if self.temp_decay != 1:
                self.temp *= self.temp_decay

            if i % self.show_every == 0:
                print('Iteration ', i)
                print('Eps: {}\nTemp: {}'.format(self.eps, self.temp))
                self.evaluate_policy()

        return self.Q, self.winrates, self.n_sub_optimals

    def withoutExploringStarts(self, numIterations=10_000):
        self.Q = self.initializeQ()
        numberOfVisits = {state: 0 for state in
                          product(range(12, 22), range(1, 11), [True, False], [0, 1])}

        for i in range(numIterations):
            state = self.env.reset()
            memory = []
            gameEnd = False

            while not gameEnd:

                if state[0] < 12:
                    newState, reward, gameEnd, _ = self.env.step(1)
                else:
                    state_values = self.Q[state]
                    action = self.explore_policy(state_values)
                    newState, reward, gameEnd, _ = self.env.step(action)
                    memory.append((state, action, reward))
                state = newState

            G = 0
            gamma = 1
            for state, action, reward in reversed(memory):
                G = gamma * G + reward
                numberOfVisits[state + (action,)] += 1
                self.Q[state][action] = self.Q[state][action] + G / numberOfVisits[state + (action,)]

            # decay eps
            if self.eps_decay != 1 and self.eps > self.min_eps:
                self.eps *= self.eps_decay
                self.eps = max(self.eps, self.min_eps)

            # temp decay
            if self.temp_decay != 1:
                self.temp *= self.temp_decay
                self.temp = max(self.temp, self.min_temp)


            if i % self.show_every == 0:
                print('Iteration ', i)
                print('Eps: {}\nTemp: {}'.format(self.eps, self.temp))
                self.evaluate_policy()

        return self.Q, self.winrates, self.n_sub_optimals

    def get_best_policy(self):
        return {state: int(values[1] > values[0]) for state, values in self.Q.items()}

    def evaluate_policy(self):
        results = {-1: 0, 0: 0, 1: 0, 1.5: 0}
        policy = self.get_best_policy()
        game = blackjack.BlackjackEnv(natural=self.natural)
        for i in range(self.evaluate_iter):
            state = game.reset()
            done = False

            while not done:
                if state[0] < 12:
                    new_state, reward, done, _ = game.step(1)
                else:
                    action = policy[state]
                    new_state, reward, done, _ = game.step(action)
                state = new_state
            results[reward] += 1

        winrate = (results[1] + results[1.5]) / self.evaluate_iter * 100
        print('Win Rate: {:.2f} % ({} games)'.format(winrate, self.evaluate_iter))
        n_sub_optimal = visualization.compare2Optimal(policy)
        print('Suboptimal Actions: {}/200\n'.format(n_sub_optimal))
        self.winrates.append(winrate)
        self.n_sub_optimals.append(n_sub_optimal)


def main():
    tic = time.time()
    # Q, winrates, n_sub_optimals = QLearning(eps=0.05, step_size=0.1, niter=100000, natural=False)
    agent = MCLearningAgent(explore_policy='decay_epsilon', eps=0.1)
    print(agent.explore_policy)
    Q, winrates, n_sub_optimals = agent.exploringStarts(numIterations=100_000_000)
    toc = time.time()
    print('Elapsed time: {:.4f} s'.format(toc - tic))
    policy = agent.get_best_policy()
    with plt.style.context('grayscale'):
        fig_policy = visualization.showPolicy(Q, policy)

    with plt.style.context('ggplot'):
        fig_learn = visualization.LearningProgess(winrates, n_sub_optimals)
    # plt.style.use('seaborn')
    plt.tight_layout()
    plt.show()

def TestParametersWES(niter=10_000, natural=False):
    step_sizes = [0.1 * i for i in range(1, 10)]
    results = {'constant_epsilon': {}, 'decay_epsilon': {}, 'boltzmann_exploration': {}}
    # Without Exploring Starts

    # Constant Epsilon
    epss = [0.05 * i for i in range(1, 4)]
    mode = 'constant_epsilon'
    for eps in epss:
        print('\n\nStarting Training for eps={:.2f}'.format(eps))
        agent = MCLearningAgent(explore_policy=mode, natural=natural, eps=eps,
                               show_every=1000,
                               evaluate_iter=10_000)
        Q, winrates, n_sub_optimals = agent.withoutExploringStarts(numIterations=niter)
        policy = agent.get_best_policy()
        results[mode][eps] = {'Q': Q, 'winrates': winrates, 'n_sub_optimals': n_sub_optimals}
        with plt.style.context('grayscale'):
            fig_policy = visualization.showPolicy(Q, policy)
            fig_policy.savefig(
                fname='graficos/MC WES__{}__{:.2f}__{}__Policy.png'.format(mode, eps, niter))
    with plt.style.context('ggplot'):
        fig_opt, fig_win = visualization.LearningProgressComparisonWithoutExploringStarts(results[mode], mode)
    fig_win.savefig('graficos/MC WES__{}__winrates.png'.format(mode))
    fig_opt.savefig('graficos/MC WES__{}__optimals.png'.format(mode))

    # Epsilon Decay
    eps_decay_rates = [0.99, 0.999, 0.9999]
    mode = 'decay_epsilon'
    for eps_decay in eps_decay_rates:
        print('\n\nStarting Training for eps_decay={:.4f} '.format(eps_decay))
        agent = MCLearningAgent(explore_policy=mode, natural=natural, eps_decay=eps_decay,
                               show_every=100_000, evaluate_iter=10_000)
        Q, winrates, n_sub_optimals = agent.withoutExploringStarts(numIterations=niter)
        policy = agent.get_best_policy()
        results[mode][eps_decay] = {'Q': Q, 'winrates': winrates, 'n_sub_optimals': n_sub_optimals}
        with plt.style.context('grayscale'):
            fig_policy = visualization.showPolicy(Q, policy)
            fig_policy.savefig(
                fname='graficos/MC WES__{}__{:.4f}__{}__Policy.png'.format(mode, eps_decay, niter))
    with plt.style.context('ggplot'):
        fig_opt, fig_win = visualization.LearningProgressComparisonWithoutExploringStarts(results[mode], mode)
    fig_win.savefig('graficos/MC WES_{}__winrates.png'.format(mode))
    fig_opt.savefig('graficos/MC WES__{}__optimals.png'.format(mode))
    temp_decay_rates = [0.99, 0.999, 0.9999]
    mode = 'boltzmann_exploration'
    for temp_decay in temp_decay_rates:
        print('\n\nStarting Training for temp_decay={:.4f}'.format(temp_decay))
        agent = MCLearningAgent(explore_policy=mode, natural=natural, temp_decay=temp_decay,
                               show_every=100_000, evaluate_iter=10_000)
        Q, winrates, n_sub_optimals = agent.withoutExploringStarts(numIterations=niter)
        policy = agent.get_best_policy()
        results[mode][(temp_decay)] = {'Q': Q, 'winrates': winrates, 'n_sub_optimals': n_sub_optimals}
        with plt.style.context('grayscale'):
            fig_policy = visualization.showPolicy(Q, policy)
            fig_policy.savefig(
                fname='graficos/MC WES__{}__{:.4f}__{}__Policy.png'.format(mode, temp_decay, niter))
    with plt.style.context('ggplot'):
        fig_opt, fig_win = visualization.LearningProgressComparisonWithoutExploringStarts(results[mode], mode)
    fig_win.savefig('graficos/MC WES__{}__winrates.png'.format(mode))
    fig_opt.savefig('graficos/MC WES__{}__optimals.png'.format(mode))
    return results

def FinalTestsES(niter=10_000_000, runs=5, natural=False, eval_games=10_000):

    f = open('MC ES_Final_Tests__{}x{}.txt'.format(runs, niter), 'w')
    f.write('Mode,Parameter,Winrate,Suboptimal Actions\n')

    agent = MCLearningAgent(natural=natural,
                            show_every=100_000, evaluate_iter=eval_games)
    winrates = []
    n_sub_optimals = []
    for i in range(1, runs + 1):
        print('Starting Run {} '.format(i))
        _, win, opt = agent.ExploringStarts(numIterations=niter)
        winrates.extend(win)
        n_sub_optimals.extend(opt)

    f.write('{},{}\n'.format(np.mean(winrates), np.mean(n_sub_optimals)))
    f.close()

def FinalTests(niter=10_000_000, runs=5, natural=False, eval_games=10_000):

    mode_params = {'constant_epsilon': [0.05, 0.10, 0.15],
                   'decay_epsilon': [0.99, 0.999, 0.9999],
                   'boltzmann_exploration': [0.99, 0.999, 0.9999]}
    f = open('MC WES_Final_Tests__{}x{}.txt'.format(runs, niter), 'w')
    f.write('Mode,Parameter,Winrate,Suboptimal Actions\n')
    for mode, values in mode_params.items():
        for val in values:
            if mode == 'constant_epsilon':
                agent = MCLearningAgent(explore_policy=mode, natural=natural, eps=val,
                                       show_every=100_000, evaluate_iter=eval_games)
            elif mode == 'decay_epsilon':
                agent = MCLearningAgent(explore_policy=mode, natural=natural, eps_decay=val,
                                       show_every=100_000, evaluate_iter=eval_games)
            else:
                agent = MCLearningAgent(explore_policy=mode, natural=natural, temp_decay=val,
                                       show_every=100_000, evaluate_iter=eval_games)

            winrates = []
            n_sub_optimals = []
            for i in range(1, runs + 1):
                print('Starting Run {} for {}, {}'.format(i, mode, val))
                _, win, opt = agent.withoutExploringStarts(numIterations=niter)
                winrates.extend(win)
                n_sub_optimals.extend(opt)

            f.write('{},{},{},{}\n'.format(mode, val, np.mean(winrates), np.mean(n_sub_optimals)))
    f.close()

if __name__ == '__main__':
    main() # MC ES
    # results = TestParametersWES()
    # FinalTests() #MC WES