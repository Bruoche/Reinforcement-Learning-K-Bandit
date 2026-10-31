import random
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable

if __name__ == "__main__":
	random.seed(67)
	np.random.seed(67)

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
		explore = random.random() < self.random_action_chance
		if explore: return self._random_action()
		else: return self._greedy_action()

	def _greedy_action(self) -> int:
		# debug("<< greeding >>")
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

def graph_reward(eps = 0.1, nb_turns = 1000, nb_players = 2000):
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


class Graph:
	def __init__(self, 
		ylabel: str,
		nb_player_types: int, 
		process_player_turn_val: Callable[[float, int, float, Ban10], float], 
		process_turn_final_val = lambda value, nb_players: value/nb_players, 
		initial_val = 0
	):
		self.ylabel = ylabel
		self.nb_player_types = nb_player_types
		self.data_per_turn_per_player_type = [DataByTurn() for _ in range(nb_player_types)]
		self.initial_val = initial_val
		self.turn_values_per_types = []
		self.nb_turn_per_types = []
		self.process_player_turn = process_player_turn_val
		self.end_turn = process_turn_final_val

	def reset_values(self):
		self.turn_values_per_types = [self.initial_val for _ in range(self.nb_player_types)]
		self.nb_turn_per_types = [0 for _ in range(self.nb_player_types)]

	def process_turn(self, player_type: int, bandit_choosen: int, reward: float, bandits: Ban10):
		self.turn_values_per_types[player_type] = self.process_player_turn(self.turn_values_per_types[player_type], bandit_choosen, reward, bandits)
		self.nb_turn_per_types[player_type] += 1

	def process_final_value(self, player_type: int):
		self.data_per_turn_per_player_type[player_type].append(self.end_turn(self.turn_values_per_types[player_type], self.nb_turn_per_types[player_type]))

	def x(self, player_type: int) -> list[int]:
		return self.data_per_turn_per_player_type[player_type].turn

	def y(self, player_type: int) -> list[float]:
		return self.data_per_turn_per_player_type[player_type].data

def process_player_rewards(current_val: float, _: int, reward: float, __: Ban10) -> float:
	return current_val + reward

def process_player_optimal(current_val: float, bandit_choosen: int, _: float, bandits: Ban10) -> float:
	if bandit_choosen == bandits.best_bandit:
		return current_val + 1
	return current_val

def all_graphs(player_constructors: Callable[[], GreedyPlayer], nb_turns = 1000, nb_players = 2000):
	bandits = Ban10()
	player_types = [[player_constructor() for _ in range(nb_players)] for player_constructor in player_constructors]
	graphs = [
		Graph("Récompense moyenne", nb_players, process_player_rewards),
		Graph("Choix optimaux (%)", nb_players, process_player_optimal)
	]
	for _ in range(nb_turns):
		for i, players in enumerate(player_types):
			for graph in graphs:
				graph.reset_values()
			for player in players:
				action = player.get_action()
				reward = bandits.play(action).item()
				player.reward(action, reward)
				for graph in graphs:
					graph.process_turn(i, action, reward, bandits)
			for graph in graphs:
				graph.process_final_value(i)
	for graph in graphs:
		plt.figure()
		labels = []
		for i, player_type in enumerate(player_types):
			labels.append(f"{player_type[0].__class__.__name__} (epsilon = {player_type[0].random_action_chance})")
			plt.plot(graph.x(i), graph.y(i), label=labels[i])
		plt.xlabel("Tour")
		plt.ylabel(graph.ylabel)
		plt.legend()
		plt.show()

all_graphs([lambda: GreedyPlayer(0), lambda: GreedyPlayer(0.01), lambda: GreedyPlayer(0.1)])

# Ex5

class OptimistGreedyPlayer(GreedyPlayer):
	def __init__(self, eps = 0.1, nb_action = DEFAULT_NB_BRAS):
		super().__init__(eps, nb_action)
		self.action_values = [5 for _ in range(nb_action)]

epsilon = 0.1
# all_graphs([lambda: GreedyPlayer(epsilon), lambda: OptimistGreedyPlayer(epsilon)])