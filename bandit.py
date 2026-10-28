import random
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

def debug(message):
	if DEBUG:
		print(message)

# Ex1

class Bandit:
	def __init__(self):
		self.avg = np.random.normal(0, 1)

	def play(self, nb_tirage = 1) -> int:
		return np.random.normal(self.avg, 1, nb_tirage)

def test_bandit_distribution():
	bandit = Bandit()
	test_tirages = bandit.play(1000)
	plt.hist(test_tirages, bins=30)
	plt.show()

# test_bandit_distribution()

# Ex2

DEFAULT_NB_BRAS = 10

class Ban10:
	def __init__(self, nb_bras=DEFAULT_NB_BRAS):
		self.bandits = [Bandit() for _ in range(nb_bras)]
		self.best_bandit = max(range(nb_bras), key=lambda i: self.bandits[i].avg)
		for i, bandit in enumerate(self.bandits):
			debug(f"Bandit {i}: {bandit.avg}{" (best)" if i == self.best_bandit else ""}")
		
	def play(self, index) -> int:
		return self.bandits[index].play()

# Ex3

class GreedyPlayer:
	def __init__(self, eps, nb_action = DEFAULT_NB_BRAS):
		self.nb_action = nb_action
		self.random_action_chance = eps # epsilon: Chances to take a random action
		self.action_values = [0 for _ in range(nb_action)] # Q : Valeur estimée pour les actions
		self.action_counts = [0 for _ in range(nb_action)] # N : Nombre de fois où chaque action à été choisie

	def get_action(self) -> int:
		explore = random.uniform(0, 1) < self.random_action_chance
		if explore: return self._random_action()
		else: return self._greedy_action()

	def _greedy_action(self) -> int:
		debug("<< greeding >>")
		candidates = []
		best = 0
		for i in range(len(self.action_counts)):
			value = self._evaluate(i)
			if (i == 0) or (value > best):
				best = value
				candidates = [i]
			elif value == best:
				candidates.append(i)
		return random.choice(candidates)

	def _evaluate(self, machine) -> float:
		nb_actions = self.action_counts[machine]
		total_value = self.action_values[machine]
		if nb_actions == 0:
			return total_value
		return total_value/nb_actions

	def _random_action(self) -> int:
		debug("<< picking random >>")
		return random.randrange(len(self.action_values))

	def reward(self, action, value):
		self.action_values[action] += value
		self.action_counts[action] += 1

def test_greedy():
	bandits = Ban10()
	player = GreedyPlayer(0.1, len(bandits.bandits))
	for _ in range(1000):
		action = player.get_action()
		reward = bandits.play(action)
		player.reward(action, reward)
		print(f"Machine {action} gave {reward} : {"Optimal" if action==bandits.best_bandit else f"Should pick {bandits.best_bandit}"}")
	for i in range(len(bandits.bandits)):
		debug(f"Machine {i} is estimated at {player._evaluate(i)} (real: {bandits.bandits[i].avg}) with a total of {player.action_values[i]} gains for {player.action_counts[i]} actions")

# test_greedy()

# Ex4

class DataByTurn:
	def __init__(self):
		self.data = []
		self.turn = []

	def append(self, data):
		self.data.append(data)
		self.turn.append(len(self.turn))

	def show_graph(self, map = lambda value: value):
		data = [map(value) for value in self.data]
		plt.plot(self.turn, data)
		plt.show()

def graph_reward(nb_turns = 1000, nb_players = 2000):
	bandits = Ban10()
	players = [GreedyPlayer(0.1, len(bandits.bandits)) for _ in range(nb_players)]
	rewards_per_turn = DataByTurn()
	for _ in range(nb_turns):
		turn_rewards = []
		for player in players:
			action = player.get_action()
			reward = bandits.play(action)
			player.reward(action, reward)
			turn_rewards.append(reward)
		rewards_per_turn.append(turn_rewards)
	rewards_per_turn.show_graph(map_avg)

def map_avg(turn_values: float) -> float:
	if len(turn_values) <= 0: return 0
	total = 0
	for value in turn_values:
		total += value
	return total/len(turn_values)

# graph_reward()

def graph_optimal(nb_turns = 1000, nb_players = 2000):
	bandits = Ban10()
	players = [GreedyPlayer(0.1, len(bandits.bandits)) for _ in range(nb_players)]
	optimals_per_turn = DataByTurn()
	for _ in range(nb_turns):
		optimals = 0
		for player in players:
			action = player.get_action()
			reward = bandits.play(action)
			player.reward(action, reward)
			if action==bandits.best_bandit:
				optimals += 1
		optimals_per_turn.append(optimals/nb_players)
	optimals_per_turn.show_graph()

# graph_optimal()

# Ex5

class OptimistGreedyPlayer(GreedyPlayer):
    def __init__(self, nb_action = DEFAULT_NB_BRAS):
        super().__init__(0, nb_action)
        self.action_values = [5 for _ in range(nb_action)]

    def reward(self, action, value):
        if self.action_counts[action] == 0:
            self.action_values[action] = 0
        super().reward(action, value)

def graph_optimist(nb_turns = 1000, nb_players = 2000):
    bandits = [Ban10() for _ in range(nb_players)]
    players = [[GreedyPlayer(0.1), OptimistGreedyPlayer()] for _ in range(nb_players)]
    rewards_per_turn = [DataByTurn(), DataByTurn()]
    optimals_per_turn = [DataByTurn(), DataByTurn()]

    for _ in range(nb_turns):
        turn_rewards = [0, 0]
        turn_optimals = [0, 0]
        for bandit, pair in zip(bandits, players):
            for i, player in enumerate(pair):
                action = player.get_action()
                reward = bandit.play(action).item()
                player.reward(action, reward)
                turn_rewards[i] += reward
                if action == bandit.best_bandit:
                    turn_optimals[i] += 1
        for i in range(2):
            rewards_per_turn[i].append(turn_rewards[i] / nb_players)
            optimals_per_turn[i].append(100 * turn_optimals[i] / nb_players)

    labels = ["Epsilon-greedy (epsilon = 0.1)", "Greedy optimiste (epsilon = 0)"]
    plt.figure()
    for i in range(2):
        plt.plot(rewards_per_turn[i].turn, rewards_per_turn[i].data, label=labels[i])
    plt.xlabel("Tour")
    plt.ylabel("Récompense moyenne")
    plt.legend()

    plt.figure()
    for i in range(2):
        plt.plot(optimals_per_turn[i].turn, optimals_per_turn[i].data, label=labels[i])
    plt.xlabel("Tour")
    plt.ylabel("Choix optimaux (%)")
    plt.legend()
    plt.show()

graph_optimist()

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    graph_optimist()