import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

# strategies are encoded as a 4-level-deep binary tree
# depends on 3 last opponent's movements
# array with 15 elements
# 0, False ~ Dove; 1, True ~ Hawk
class EvolutionaryPlayer:
    # Init without given strategy creates a player with a strategy evolved with evolutionary algorithm
    def __init__(self, strategy = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1], strategyType = ""):
        if strategyType == "oppositeTFT":
            self.strategy = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        elif strategyType == "TFT":
            self.strategy = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        else:
            self.strategy = strategy
        self.moves = 3

    # Returns a player's move based on all previous opponent's moves
    def play(self, opponentMoves):
        moves = opponentMoves[(-1 * self.moves):]
        act_pos = 0
        for i in moves:
            if (i):
                act_pos = act_pos * 2 + 2
            else:
                act_pos = act_pos * 2 + 1
        return self.strategy[act_pos] == 1

    def playAxelrod(self, opponentMoves):
        moves = opponentMoves[(-1 * self.moves):]
        act_pos = 0
        for i in moves:
            if (i == 'H'):
                act_pos = act_pos * 2 + 2
            else:
                act_pos = act_pos * 2 + 1
        if self.strategy[act_pos] == 1:
             return 'H'
        return 'D'


class ChickenGame:
    def __init__(self, rounds, c, v):
        self.moves1 = []
        self.moves2 = []
        self.score1 = 0
        self.score2 = 0
        self.rounds = rounds
        self.c = c
        self.v = v

    # Calculates scores for one game
    def judge(self, move1, move2):
        if move1 and move2:
            self.score1 += (self.v - self.c)/2
            self.score2 += (self.v - self.c)/2
        elif move1 and not move2:
            self.score1 += self.v
            self.score2 += 0
        elif not move1 and move2:
            self.score1 += 0
            self.score2 += self.v
        elif not move1 and not move2:
            self.score1 += self.v/2
            self.score2 += self.v/2

    # Simulates a tournament (nr of games = self.rounds) between given players
    def play(self, player1, player2):
        self.moves1 = []
        self.moves2 = []
        self.score1 = 0
        self.score2 = 0
        for _ in range(self.rounds):
            m1 = player1.play(self.moves2)
            m2 = player2.play(self.moves1)
            self.judge(m1, m2)
            self.moves1.append(m1)
            self.moves2.append(m2)

class Algorithm:
    def __init__(self, length):
        self.population = []
        self.scores = []
        self.length = length
        self.populationsScores = []
        self.opponentScores = []
        random.seed(5001)

    # Generates population (population size = 100) of Evolutionary players with random strategies
    def generatePopulation(self):
        self.population = [EvolutionaryPlayer(random.choices([0, 1], k=self.length)) for i in range(100)]
        self.scores = [0 for i in range(100)]
        self.populationsScores = []
        self.opponentScores = []

    # Makes 10 confrontations for each player, each confrontation consists of 60 games
    def confrontation(self):
        plen = len(self.population)

        # Makes 10 games with random opponent for each player
        games = list(range(plen))*10
        random.shuffle(games)
        games = list(zip(*[iter(games)]*2))

        # Run all games and calculate total score for each player
        game = ChickenGame(60, 6, 4)
        for (i, j) in games:
            game.play(self.population[i], self.population[j])
            self.scores[i] += game.score1
            self.scores[j] += game.score2
        self.populationsScores.append(np.mean(self.scores))
    
    # Makes confrontation for each player with opposite TFT, each confrontation consist of 100 games
    def confrontation_with_OppositeTFT(self):
        plen = len(self.population)

        # Run all games and calculate total score for each player
        opptftScores = []
        game = ChickenGame(100, 6, 4)
        for i in range(plen):
            # player with encoded strategy for opposite tft
            player = EvolutionaryPlayer(strategyType="oppositeTFT")
            game.play(self.population[i], player)
            self.scores[i] += game.score1
            opptftScores.append(game.score2)

        self.populationsScores.append(np.mean(self.scores))
        self.opponentScores.append(np.mean(opptftScores))
    
    # Makes confrontation for each player with TFT, each confrontation consist of 100 games
    def confrontation_with_TFT(self):
        plen = len(self.population)

        # Run all games and calculate total score for each player
        tftScores = []
        game = ChickenGame(100, 6, 4)
        for i in range(plen):
            # player with encoded strategy for tft
            player = EvolutionaryPlayer(strategyType="TFT")
            game.play(self.population[i], player)
            self.scores[i] += game.score1
            tftScores.append(game.score2)

        self.populationsScores.append(np.mean(self.scores))
        self.opponentScores.append(np.mean(tftScores))


    # Makes a new population consists of 20 best games in five copies
    def bestPlayers(self):
        best20 = sorted(range(len(self.scores)), key = lambda x: self.scores[x])[-20:]
        newPopulation = [self.population[i] for i in best20]*5
        random.shuffle(newPopulation)
        self.population = newPopulation
        self.scores = [0 for i in range(100)]

    # One-point crossover - with probablity equals to 0.7
    def crossover(self):
        offspring = []
        for (i, j) in list(zip(*[iter(self.population)]*2)):
            strategy1 = i.strategy
            strategy2 = j.strategy
            crossPoint = 7
            crossProb = 0.7
            crossover = random.choices([True, False], [crossProb, 1-crossProb], k=1)
            if (crossover[0]):
                new1 = EvolutionaryPlayer(strategy1[:crossPoint] + strategy2[crossPoint:])
                new2 = EvolutionaryPlayer(strategy2[:crossPoint] + strategy1[crossPoint:])
                offspring.append(new1)
                offspring.append(new2)
            else:
                offspring.append(i)
                offspring.append(j)
        self.population = offspring

    # Bitflip mutation (fixed probability 1/15 at each bit)
    def mutation(self):
        mutProb = 1/(self.length)
        for player in self.population:
            mutations = random.choices([True, False], [mutProb, 1-mutProb], k=self.length)
            strategy = player.strategy
            for i in range(self.length):
                if (mutations[i]):
                    strategy[i] = 1 - strategy[i]

    # Return the best player after 10000 iterations
    def run(self):
        self.generatePopulation()
        
        for _ in range(10000):
            self.confrontation()
            self.bestPlayers()
            self.crossover()
            self.mutation()

        self.confrontation()

        bestId = sorted(range(len(self.scores)), key = lambda x: self.scores[x])[-1]
        return self.population[bestId]

    def run_with_oppositeTFT(self):
        self.generatePopulation()
        
        for i in range(1000):
            print(i)
            self.confrontation_with_OppositeTFT()
            self.bestPlayers()
            self.crossover()
            self.mutation()

        self.confrontation_with_OppositeTFT()

        bestId = sorted(range(len(self.scores)), key = lambda x: self.scores[x])[-1]
        return self.population[bestId]
    
    def run_with_TFT(self):
        self.generatePopulation()
        
        for i in range(100):
            self.confrontation_with_TFT()
            self.bestPlayers()
            self.crossover()
            self.mutation()

        self.confrontation_with_TFT()

        bestId = sorted(range(len(self.scores)), key = lambda x: self.scores[x])[-1]
        return self.population[bestId]

        
# Method used to create evolutionary evolved player
def runAlg():
    alg = Algorithm(15)
    evolutionary = alg.run()
    print(evolutionary.strategy) # [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1]
    plt.plot(alg.populationsScores)
    plt.savefig('populationScores.png')
    plt.show()

def runOppositeTFT():
    alg = Algorithm(15)
    f, axs = plt.subplots(5, 2, figsize=(15, 7))
    for i in range(5):
        for j in  range(2):
            evolutionary = alg.run_with_oppositeTFT()
            axs[i, j].plot(alg.populationsScores, label="evolutionary")
            axs[i, j].plot(alg.opponentScores, label="OppositeTFT")
        
    handles, labels = axs[0, 0].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center')
    plt.subplots_adjust(hspace=1, wspace=0.5)
    f.savefig('populationScoresOppTFT.png')

def runTFT():
    alg = Algorithm(15)
    f, axs = plt.subplots(5, 1, figsize=(15, 7))
    for i in range(5):
        evolutionary = alg.run_with_TFT()
        axs[i].plot(alg.populationsScores, label="evolutionary")
        axs[i].plot(alg.opponentScores, label="TFT")
        print(alg.opponentScores)
        
    handles, labels = axs[0].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center')
    plt.subplots_adjust(hspace=2, wspace=1)
    f.savefig('populationScoresTFT.png')


