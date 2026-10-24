import random
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

def debug(message):
	if DEBUG:
		print(message)

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

test_bandit_distribution()

DEFAULT_NB_BRAS = 10

class Ban10:
	def __init__(self, nb_bras=DEFAULT_NB_BRAS):
		self.bandits = [Bandit() for _ in range(nb_bras)]
		self.best_bandit = max(range(nb_bras), key=lambda i: self.bandits[i].avg)
		for i, bandit in enumerate(self.bandits):
			debug(f"Bandit {i}: {bandit.avg}{" (best)" if i == self.best_bandit else ""}")
		
	def play(self, index) -> int:
		return self.bandits[index].play()

class GreedyPlayer:
	def __init__(self, eps, nb_action = DEFAULT_NB_BRAS):
		self.nb_action = nb_action
		self.random_action_chance = eps # epsilon: Chances to take a random action
		self.action_values = [0 for _ in range(nb_action)] # Q : Valeur estimée pour les actions
		self.action_counts = [0 for _ in range(nb_action)] # N : Nombre de fois où chaque action à été choisie

	def get_action(self) -> int:
		explore = random.uniform(0, 1) < self.random_action_chance
		if explore:
			return self._random_action()
		else:
			return self._greedy_action()

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

def main():
	bandits = Ban10()
	player = GreedyPlayer(0.1, len(bandits.bandits))
	for _ in range(1000):
		action = player.get_action()
		reward = bandits.play(action)
		player.reward(action, reward)
		print(f"Machine {action} gave {reward} : {"Optimal" if action==bandits.best_bandit else f"Should pick {bandits.best_bandit}"}")
	for i in range(len(bandits.bandits)):
		debug(f"Machine {i} is estimated at {player._evaluate(i)} (real: {bandits.bandits[i].avg}) with a total of {player.action_values[i]} gains for {player.action_counts[i]} actions")

# main()