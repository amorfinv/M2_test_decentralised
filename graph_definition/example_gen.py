# genetic algorithm example

import pandas as pd
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools

#goal percentages
total_calories = 2500 * 7
percentage_prot = 0.3
percentage_carb = 0.5
percentage_fat = 0.2

# compute total calories per macro
cal_prot = round(percentage_prot * total_calories)
cal_carb = round(percentage_carb * total_calories)
cal_fat = round(percentage_fat * total_calories)
print(cal_prot, cal_carb, cal_fat)

# fixed info on macro nutriments: calories per gram of protein, carb and fat
prot_cal_p_gram = 4
carb_cal_p_gram = 4
fat_cal_p_gram = 9

#goal grams
gram_prot = cal_prot / prot_cal_p_gram
gram_carb = cal_carb / carb_cal_p_gram
gram_fat = cal_fat / fat_cal_p_gram
print(gram_prot, gram_carb, gram_fat)

# per week: min, max, cal unit, prot g,  fat g, carb g
products_table = pd.DataFrame.from_records([
    ['Banana 1u', 0, 4, 89, 1, 0, 23],
    ['Mandarin 1u', 0, 4, 40, 1, 0, 10],
    ['Ananas 100g', 0, 7, 50, 1, 0, 13],
    ['Grapes 100g', 0, 7, 76, 1, 0, 17],
    ['Chocolate 1 bar', 0, 4, 230, 3, 13, 25],
    
    ['Hard Cheese 100g', 0, 8, 350, 28, 26, 2],
    ['Soft Cheese 100g', 0, 8, 374, 18, 33, 1],
    ['Pesto 100g', 0, 8, 303, 3, 30, 4],
    ['Hoummous 100g', 0, 8, 306, 7, 25, 11],
    ['Aubergine Paste 100g', 0, 4, 228, 1, 20, 8],
    
    ['Protein Shake', 0, 5, 160, 30, 3, 5],
    ['Veggie Burger 1', 0, 5, 220, 21, 12, 3],
    ['Veggie Burger 2', 0, 12, 165, 16, 9, 2],
    ['Boiled Egg', 0, 8, 155, 13, 11, 1],
    ['Backed Egg', 0, 16, 196, 14, 15, 1],
    
    ['Baguette Bread Half', 0, 3, 274, 10, 0, 52],
    ['Square Bread 1 slice', 0, 3, 97, 3, 1, 17],
    ['Cheese Pizza 1u', 0, 3, 903, 36, 47, 81],
    ['Veggie Pizza 1u', 0, 3, 766, 26, 35, 85],
    
    ['Soy Milk 200ml', 0, 1, 115, 8, 4, 11],
    ['Soy Chocolate Milk 250ml', 0, 3, 160, 7, 6,20],
    
])
products_table.columns = ['Name', 'Min', 'Max', 'Calories', 'Gram_Prot', 'Gram_Fat', 'Gram_Carb']

products_table

# extract the information of products in a format that is easier to use in the deap algorithms cost function
cal_data = products_table[['Gram_Prot', 'Gram_Fat', 'Gram_Carb']]

prot_data = list(cal_data['Gram_Prot'])
fat_data = list(cal_data['Gram_Fat'])
carb_data = list(cal_data['Gram_Carb'])

# the random initialization of the genetic algorithm is done here
# it gives a list of integers with for each products the number of times it is bought
def n_per_product():
    return random.choices( range(0, 10), k = 21)

# this is the function used by the algorithm for evaluation
# I chose it to be the absolute difference of the number of calories in the planning and the goal of calories
def evaluate(individual):
    individual = individual[0]
    tot_prot = sum(x*y for x,y in zip(prot_data,individual))
    tot_fat = sum(x*y for x,y in zip(fat_data,individual))
    tot_carb = sum(x*y for x,y in zip(carb_data,individual))
    cals = prot_cal_p_gram * tot_prot + carb_cal_p_gram * tot_carb + fat_cal_p_gram * tot_fat
    return abs(cals - total_calories),

# this is the setup of the deap library: registering the different function into the toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("n_per_product", n_per_product)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.n_per_product, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.population(n=10)

# this is the definition of the total genetic algorithm is executed, it is almost literally copied from the deap library
def main():
    pop = toolbox.population(n=4)
    print(pop)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    print(fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    
    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < 5000:
        # A new generation
        g = g + 1
        #print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values
            
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        #print(min(fits), max(fits), mean, std)
    
    best = pop[np.argmin([toolbox.evaluate(x) for x in pop])]
    return best

if __name__ == '__main__':
    best_solution = main()

    products_table['univariate_choice'] = pd.Series(best_solution[0])
    print(products_table.head())