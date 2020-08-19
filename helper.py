import evolutionaryAlgorithm as ea 

# Tak tworzymy gracza
player = ea.EvolutionaryPlayer()

# Tu są przykładowe ruchy przeciwnika (wszystkie do tej pory w poprzednich rozgrywkach)
# True -> Hawk, False -> Dove
opponents = [True, False]

# Ta funkcja zwraca aktualny ruch gracza
player.play(opponents)