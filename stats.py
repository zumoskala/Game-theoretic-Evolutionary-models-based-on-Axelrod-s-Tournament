# histogram wszystkich końcowych wyników
# wykres przebiegu rozgrywki

# imports
import Game as game
import seaborn as sns
import numpy as np

#############################
# DATA FRAME FOR STATISTICS #
#############################
game_stats = []
for elt in game.duel_sorted:
    game_list = []
    game_list.append(elt[0][0])
    game_list.append(elt[0][1])
    game_list.append(sum(elt[1]))
    game_list.append(np.array(elt[1]))
    game_stats.append(game_list)

import pandas as pd

df = pd.DataFrame(game_stats, columns=['Player', 'Opponent', 'Score', 'List of Scores'])
#print(df)

#############################
#           PLOTS           #
#############################

import seaborn as sns
import matplotlib.pyplot as plt

# final results - heatmap
sns.set()
final_stats = df.pivot('Player', 'Opponent', 'Score')
ax = sns.heatmap(final_stats, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
plt.show()

# final results - histogram for every player vs opponents
results_together = sns.barplot(x="Player", y="Score", hue="Opponent", data=df, palette="GnBu_d")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
results_together.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.show()

ea = df[df['Player'] == 'EvolutionaryAlg']

def sumScores(list):
    sscores = []
    current = list[0]
    for i in range(len(list)):
        sscores.append(current)
        current+=list[i]
    return sscores

eavsallc = sumScores(ea.values[0][3])
eavsalnc = sumScores(ea.values[1][3])
eavsatft = sumScores(ea.values[2][3])
eavsea = sumScores(ea.values[3][3])
eavsgr = sumScores(ea.values[4][3])
eavsotft = sumScores(ea.values[5][3])
eavsrand = sumScores(ea.values[6][3])
eavstft = sumScores(ea.values[7][3])

plt.plot(eavsallc, color='green', label = 'EA vs ALLC')
plt.plot(eavsalnc, color='purple', label = 'EA vs ALNC')
plt.plot(eavsatft, color='yellow', label = 'EA vs ATFT')
plt.plot(eavsea, color='red', label = 'EA vs EA')
plt.plot(eavsgr, color='pink', label = 'EA vs Grudger')
plt.plot(eavsotft, color='blue', label = 'EA vs OppositeTFT')
plt.plot(eavsrand, color='black', label = 'EA vs Random')
plt.plot(eavstft, color='cyan', label = 'EA vs TFT')
plt.legend()
plt.show()

print(sum(ea.values[1][3]))
print(sum(ea.values[4][3]))
