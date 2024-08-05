import random


class EvolutionaryCommunityDetection:
    def __init__(self, network, pop_size=10, generations=10, mutation_rate=0.1):
        self.network = network
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def compute_density_fitness(self, communities):
        """
        Fitness function based on the density of the network.

        D = 2 * internal_edges / num_nodes * (num_nodes - 1)

        :param communities: Collection of communities from the network.
        """
        density_scores = []
        for community in communities:
            num_nodes = len(community)

            # There is only one member in the community.
            if num_nodes <= 1:
                density_scores.append(0)

            # Compute the density of the community.
            else:
                num_edges = sum(
                    len(set(self.network[node]) & set(community)) for node in community
                )
                density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
                density_scores.append(density)

        # Avoid division by 0 if there are no communities.
        return sum(density_scores) / len(density_scores) if density_scores else 0

    def compute_centralization_fitness(self, communities):
        """
        Fitness function based on centralizing the nodes.

        C = sum_grades_of_all_nodes / num_nodes

        :param communities: Collection of communities from the network.
        """
        centralization_scores = []
        for community in communities:
            centralization = 0
            for node in community:
                # Degree of a node = number of direct neighbors.
                degree = sum(1 for neighbor in self.network[node])
                centralization += degree

            num_nodes = len(community)
            # Avoid divison by 0.
            centralization = centralization / num_nodes if num_nodes > 0 else 0

            centralization_scores.append(centralization)

        # Here we can avoid by default the division by 0.
        return sum(centralization_scores) / len(centralization_scores)

    def compute_modularity(self, communities):
        """
            Computes the modularity of a network.
            Modularity = the strength of division of a network between modules (groups/communities).
            
            For a community, it is defined as: sum(ratio_intra - ratio_inter**2) for all communities
                - ratio_intra = ratio of edges that connect the communities.
                - ratio_inter = ratio of edges that connect the nodes in a community.

            :param communities: Collection of communities from a network.
        """
        total_edges = sum(
            len(list(self.network.neighbors(node))) for node in self.network
        )
        
        modularity = 0
        for community in communities:
            # The number of nodes between the communities.
            intra_edges = sum(
                len(set(self.network.neighbors(node)) & set(community))
                for node in community
            )
            
            # The number of nodes between the members of the community.
            inter_edges = sum(
                len(set(self.network.neighbors(node)) - set(community))
                for node in community
            )
            
            modularity += (intra_edges / total_edges) - (inter_edges / total_edges) ** 2
        return modularity

    def combine_fitness_scores(self, scores):
        """
            Combines the fitness scores
            
            :param scores: Collection of fitness scores.
        """
        return sum(scores) / len(scores) if scores else 0

    def initialize_population(self):
        """
            Initializes a population.
            For each node, the community is chosen randomly
        """
        population = []
        nodes = list(self.network.nodes())
        for _ in range(self.pop_size):
            random.shuffle(nodes)
            individual = [
                nodes[i : i + random.randint(1, len(nodes))]
                for i in range(0, len(nodes), random.randint(1, len(nodes) // 2))
            ]
            population.append(individual)
        return population

    def selection(self, population, fitness_scores):
        """
            Selects the next node, randomly from all nodes,
            based on the fitness scores.
            
            :param population: The population initialized before.
            :param fitness_scores: The computed fitness scores.
        """
        selected = []
        for _ in range(len(population)):
            # Choosing 5 random nodes to choose from.
            participants = random.sample(list(enumerate(population)), 5)
            
            # Ordering the random chosen nodes by fitness score.
            winner = max(participants, key=lambda x: fitness_scores[x[0]])
            selected.append(winner[1])
        return selected

    def crossover(self, parent1, parent2):
        """
            Combines two cells to create a child -> combines two networks to
            encourage exploration and diversity of the network.
            
            :param parent1: Network.
            :param parent2: Network.
        """
        child = []
        for comm1, comm2 in zip(parent1, parent2):
            if random.random() > 0.5:
                child.append(comm1)
            else:
                child.append(comm2)
        return child

    def mutation(self, individual):
        """
            Mutates a nodes -> passes a node from a community to another.
            
            :param individual: Community to be mutated.
        """
        mutated = individual.copy()
        nodes = list(self.network.nodes())
        for i in range(len(mutated)):
            # Mutates if the mutation_rate is greater.
            if random.random() < self.mutation_rate:
                # Select a random node to be reassigned.
                node_to_reassign = random.choice(mutated[i])
                
                # Assign the node to a random community.
                mutated[i] = [node for node in mutated[i] if node != node_to_reassign]
                random_community = random.choice(mutated)
                random_community.append(node_to_reassign)
        return mutated

    def run(self):
        # Initializes the population.
        population = (self.initialize_population())
        
        # Running the evolution on generations.
        for gen in range(self.generations):
            # Computing the modularity and density of the network.
            
            # How well divised is the graph?
            modularity_scores = [self.compute_modularity(individual) for individual in population]
            
            # How dense is each community?
            density_scores = [self.compute_density_fitness(individual) for individual in population]
            
            # Computing the centralization scores.
            centralization_scores = [self.compute_centralization_fitness(individual) for individual in population]

            # Combining all the scores.
            combined_scores = [(modularity_scores[i], density_scores[i], centralization_scores[i]) for i in range(len(population))]

            # Compute the overall fitness for each individual.
            overall_fitness = [self.combine_fitness_scores(scores) for scores in combined_scores]

            # Retrieving the overall best solution from a generation.
            best_index = overall_fitness.index(max(overall_fitness))
            best_solution = population[best_index]
            
            print(f"Generation {gen + 1}: Best Modularity = {modularity_scores[best_index]}")

        return best_solution
    
