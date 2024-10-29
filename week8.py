import random
target = "I am the right answer! How many generations?  Genetic Algorithm."
gene_pool = "abcdefghiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.!? "
population_size = 10

population = []
score_record = []

def calculate_score(chromosome):
	score = 0
	for i in range(len(chromosome)):
		if chromosome[i] == target[i]:
			score += 1
	return score

def crossover(parentA, parentB):
	#1_half
	crossover_point=int(len(parentA)/2)
	#print(crossover_point)
	#2_random point
	#crossover_point=random.randrange(len(parentA))
	child1 = parentA[:crossover_point] + parentB[crossover_point:]
	child2 = parentB[:crossover_point] + parentA[crossover_point:]
	#print(len(child))
	return child1,child2

def two_point_crossover(parentA, parentB):
	point_one=random.randrange(len(parentA))
	point_two=random.randrange(len(parentA))
	if point_one > point_two:
		temp = point_two
		point_two = point_one
		point_one = temp
	child1 = parentA[:point_one] + parentB[point_one:point_two] + parentA[point_two:]
	child2 = parentB[:point_one] + parentA[point_one:point_two] + parentB[point_two:]
	return child1,child2
		
def mutate(chromosome):
	for i in range(2):
		mutate_point=random.randrange(len(chromosome))
		chromosome=list(chromosome)
		chromosome[mutate_point]=random.choice(list(gene_pool))
		chromosome= "".join(chromosome)
	return chromosome

#initialize population and scores
for i in range(population_size):
	chromosome = ""
	for j in range(len(target)):
		chromosome += random.choice(list(target))
	population.append(chromosome)
	score_record.append(calculate_score(chromosome))

# evolution
found = False
for i in range(20000):
	print("Generation ", i, ":")
	# select parents
	#1_wolf? only the two with max score will produce
	#2_lion? the one with highest * other
	#3 higher with higher, lower with lower
	#4 higher with lower
	#5 random
	for j in range(population_size):
		m=random.randrange(population_size)
		n=random.randrange(population_size)
		child1,child2=two_point_crossover(population[m],population[n])
		population.append(child1)
		population.append(child2)
		score_record.append(calculate_score(child1))
		score_record.append(calculate_score(child2))
		if child1 == target or child2 == target:
			found = True
			break
	if found:
		break
	while len(population) != population_size:
		to_delete_index = score_record.index(min(score_record))
		del population[to_delete_index]
		del score_record[to_delete_index]
	for j in range(population_size):
		if random.randrange(10) < 2:
			mutated = mutate(population[j])
			population[j]=mutated
			score_record[j]=calculate_score(mutated)
			if mutated == target:
				found = True
				break
	if found:
		break

	print(population)
	print(score_record)

while len(population) != population_size:
		to_delete_index = score_record.index(min(score_record))
		del population[to_delete_index]
		del score_record[to_delete_index]
print("Found in generation", i)
print(population)
print(score_record)






