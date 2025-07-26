import random

items = [
    {"name":"Item-1", "weight":7, "value": 8},
    {"name":"Item-2", "weight":8, "value": 10},
    {"name":"Item-3", "weight":5, "value": 6},
    {"name":"Item-4", "weight":1, "value": 2},
    {"name":"Item-5", "weight":3, "value": 4},
    {"name":"Item-6", "weight":6, "value": 7},
    {"name":"Item-7", "weight":2, "value": 4},
]

knapsack_capacity = 25

population_size=100
mutation_rate=0.1
num_generations = 100

def create_individual():
    return [random.randint(0,1) for i in range(len(items))]

def create_population():
    return [create_individual() for i in range(population_size)]

def evaluate_fitness(individual):
    total_weight = 0
    total_value = 0
    for i in range(len(items)):
        if individual[i] == 1:
            total_weight += items[i]["weight"]
            total_value += items[i]["value"]
    if total_weight > knapsack_capacity:
        total_value=0
    return total_value

def select_parents(population):
    fitness_values = [evaluate_fitness(individual) for individual in population]
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return random.sample(population, 2)
    probabilities = [fitness/total_fitness for fitness in fitness_values]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents[0], parents[1]

def crossover(parent1, parent2):
    crossover_point=random.randint(1,len(parent1)-1)
    child1=parent1[:crossover_point]+ parent2[crossover_point:]
    child2=parent2[:crossover_point]+ parent1[crossover_point:]

    return child1, child2 

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i]=1-individual[i]

def genetic_algorithm():
    population=create_population()
    for _generation in range(num_generations):
        new_population=[]
    
        for _ in range(population_size//2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1,child2])
        population=new_population
    best_individual=max(population,key=evaluate_fitness)
    best_fitness=evaluate_fitness(best_individual)
    return best_individual,best_fitness

best_solution , best_fitness = genetic_algorithm()
print("Best solution: ", best_solution)
print("Best fitness: ", best_fitness)
