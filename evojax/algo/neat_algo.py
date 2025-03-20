"""Implementation of the NEAT algorithm.

NEAT (NeuroEvolution of Augmenting Topologies) evolves both the weights
and the structure of neural networks. This algorithm starts with minimal
network topologies and gradually adds complexity through mutations.

Key features of NEAT:
- Simultaneous evolution of network structure and connection weights
- Protection of innovation through speciation (grouping similar networks)
- Minimizing dimensionality by starting from minimal structures
- Historical markings to enable meaningful crossover between different topologies

This implementation provides:
- Support for feed-forward neural network topologies
- Speciation based on topological similarity
- Various mutation operators (add/remove nodes, add/remove connections, weight mutation)
- Configuration parameters to control the evolutionary process
- Compatibility with the EvoJAX framework for JAX-accelerated evolution
"""

import logging
import numpy as np
from typing import Union, List

import jax
import jax.numpy as jnp

from evojax.algo.base import NEAlgorithm
from evojax.util import create_logger


class NEATConfig:
    """Configuration for NEAT algorithm."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_nodes: int = 30,
        max_connections: int = 100,
        compatibility_threshold: float = 3.8,
        compatibility_disjoint_coefficient: float = 1.0,
        compatibility_weight_coefficient: float = 0.6,
        conn_add_prob: float = 0.15,
        conn_delete_prob: float = 0.08,
        node_add_prob: float = 0.1,
        node_delete_prob: float = 0.03,
        act_fn_mutate_prod: float = 0.1,
        weight_mutate_prob: float = 0.9,
        weight_mutate_power: float = 2.0,
        survival_threshold: float = 0.3,
        elitism: int = 3,
    ):
        """Initialize NEAT configuration.

        Args:
            input_dim: Number of input nodes
            output_dim: Number of output nodes
            max_nodes: Maximum number of nodes
            max_connections: Maximum number of connections
            compatibility_threshold: Threshold for speciation
            compatibility_disjoint_coefficient: Weight for disjoint genes in compatibility
            compatibility_weight_coefficient: Weight for connection weights in compatibility
            conn_add_prob: Probability of adding a connection
            conn_delete_prob: Probability of deleting a connection
            node_add_prob: Probability of adding a node
            node_delete_prob: Probability of deleting a node
            act_fn_mutate_prod: Probability of mutating activation functions
            weight_mutate_prob: Probability of mutating connection weights
            weight_mutate_power: Power of weight mutation
            survival_threshold: Fraction of each species to keep for reproduction
            elitism: Number of top genomes to copy to next generation unchanged
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        self.max_connections = max_connections
        self.compatibility_threshold = compatibility_threshold
        self.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
        self.compatibility_weight_coefficient = compatibility_weight_coefficient
        self.conn_add_prob = conn_add_prob
        self.conn_delete_prob = conn_delete_prob
        self.node_add_prob = node_add_prob
        self.node_delete_prob = node_delete_prob
        self.act_fn_mutate_prod = act_fn_mutate_prod
        self.weight_mutate_prob = weight_mutate_prob
        self.weight_mutate_power = weight_mutate_power
        self.survival_threshold = survival_threshold
        self.elitism = elitism

        # Initial number of nodes: inputs + outputs + bias
        self.initial_nodes = input_dim + output_dim + 1
        self.bias_node_id = input_dim
        self.output_start_id = input_dim + 1


class Species:
    """Species class for NEAT."""

    def __init__(self, id: int, representative: np.ndarray):
        """Initialize a species.

        Args:
            id: Species ID
            representative: Representative genome for this species
        """
        self.id = id
        self.representative = representative
        self.members = []
        self.fitness_history = []
        self.last_improved = 0
        self.best_fitness = -float("inf")
        self.adjusted_fitness_sum = 0.0


class NEAT(NEAlgorithm):
    """Implementation of NEAT algorithm."""

    def __init__(
        self,
        param_size: int,
        pop_size: int,
        input_dim: int,
        output_dim: int,
        max_nodes: int = 30,
        max_connections: int = 100,
        init_stdev: float = 0.5,
        seed: int = 0,
        logger: logging.Logger = None,
        compatibility_threshold: float = 3.8,
        compatibility_disjoint_coefficient: float = 1.0,
        compatibility_weight_coefficient: float = 0.6,
        conn_add_prob: float = 0.15,
        conn_delete_prob: float = 0.08,
        node_add_prob: float = 0.1,
        node_delete_prob: float = 0.03,
        act_fn_mutate_prod: float = 0.1,
        weight_mutate_prob: float = 0.9,
        weight_mutate_power: float = 2.0,
        survival_threshold: float = 0.3,
        elitism: int = 3,
    ):
        """Initialize NEAT.

        Args:
            param_size: Size of parameter vector
            pop_size: Population size
            input_dim: Number of input nodes
            output_dim: Number of output nodes
            max_nodes: Maximum number of nodes
            max_connections: Maximum number of connections
            init_stdev: Initial standard deviation for weights
            seed: Random seed
            logger: Logger
            compatibility_threshold: Threshold for speciation
            compatibility_disjoint_coefficient: Weight for disjoint genes in compatibility
            compatibility_weight_coefficient: Weight for connection weights in compatibility
            conn_add_prob: Probability of adding a connection
            conn_delete_prob: Probability of deleting a connection
            node_add_prob: Probability of adding a node
            node_delete_prob: Probability of deleting a node
            act_fn_mutate_prod: Probability of mutating activation functions
            weight_mutate_prob: Probability of mutating connection weights
            weight_mutate_power: Power of weight mutation
            survival_threshold: Fraction of each species to keep for reproduction
            elitism: Number of top genomes to copy to next generation unchanged
        """
        if logger is None:
            self.logger = create_logger(name="NEAT")
        else:
            self.logger = logger

        self.pop_size = pop_size
        self.param_size = param_size
        self.rng = np.random.RandomState(seed)
        self.generation = 0

        # Configuration
        self.config = NEATConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            max_nodes=max_nodes,
            max_connections=max_connections,
            compatibility_threshold=compatibility_threshold,
            compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
            compatibility_weight_coefficient=compatibility_weight_coefficient,
            conn_add_prob=conn_add_prob,
            conn_delete_prob=conn_delete_prob,
            node_add_prob=node_add_prob,
            node_delete_prob=node_delete_prob,
            act_fn_mutate_prod=act_fn_mutate_prod,
            weight_mutate_prob=weight_mutate_prob,
            weight_mutate_power=weight_mutate_power,
            survival_threshold=survival_threshold,
            elitism=elitism,
        )

        # Track species
        self.species = {}
        self.species_counter = 0

        # Initialize population
        self.population = []
        self.current_node_id = self.config.initial_nodes
        self._initialize_population(init_stdev)

        # Best params tracking
        self._best_params = None
        self._best_fitness = -float("inf")

        # JAX functions
        self.jnp_array = jax.jit(jnp.array)
        self.jnp_stack = jax.jit(jnp.stack)

    def _initialize_population(self, init_stdev: float) -> None:
        """Initialize population with minimal networks.

        Args:
            init_stdev: Initial standard deviation for weights
        """
        self.logger.info(f"Initializing population with size {self.pop_size}")

        for _ in range(self.pop_size):
            # Create a minimal network with direct connections from inputs to outputs
            genome = self._create_initial_genome(init_stdev)
            self.population.append(genome)

    def _create_initial_genome(self, init_stdev: float) -> np.ndarray:
        """Create an initial genome with minimal structure.

        Args:
            init_stdev: Initial standard deviation for weights

        Returns:
            Genome as a flat numpy array
        """
        genome = np.zeros(self.param_size)

        # Set up nodes section first
        node_section = genome[: 3 * self.config.max_nodes].reshape(
            self.config.max_nodes, 3
        )

        # Set inputs, bias, and output nodes
        for i in range(self.config.input_dim):
            # Change: Assign random activation function
            node_section[i] = [
                i,
                0,
                self.rng.randint(0, 9),
            ]  # id, type=input, random activation

        # Set bias node
        node_section[self.config.bias_node_id] = [
            self.config.bias_node_id,
            0,
            self.rng.randint(0, 9),
        ]

        # Set output nodes
        for i in range(self.config.output_dim):
            node_id = self.config.output_start_id + i
            node_section[node_id] = [
                node_id,
                1,
                self.rng.randint(0, 9),
            ]  # random activation

        # Now set up initial connections from each input to each output
        conn_section = genome[3 * self.config.max_nodes :].reshape(
            self.config.max_connections, 4
        )
        conn_idx = 0

        # Connect inputs to outputs
        for i in range(self.config.input_dim):
            for j in range(self.config.output_dim):
                to_node = self.config.output_start_id + j
                conn_section[conn_idx] = [i, to_node, self.rng.normal(0, init_stdev), 1]
                conn_idx += 1

        # Connect bias to outputs
        for j in range(self.config.output_dim):
            to_node = self.config.output_start_id + j
            conn_section[conn_idx] = [
                self.config.bias_node_id,
                to_node,
                self.rng.normal(0, init_stdev),
                1,
            ]
            conn_idx += 1

        return genome

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Mutate a genome.

        Args:
            genome: Genome to mutate

        Returns:
            Mutated genome
        """
        # Make a copy of the genome
        new_genome = genome.copy()

        # Extract node and connection sections
        node_section = new_genome[: 3 * self.config.max_nodes].reshape(
            self.config.max_nodes, 3
        )
        conn_section = new_genome[3 * self.config.max_nodes :].reshape(
            self.config.max_connections, 4
        )

        # Get active nodes and connections
        active_nodes = self._get_active_nodes(node_section)
        active_conns = self._get_active_connections(conn_section)

        # Possibly add a node
        if (
            self.rng.random() < self.config.node_add_prob
            and len(active_nodes) < self.config.max_nodes
            and active_conns
        ):
            self._mutate_add_node(node_section, conn_section, active_conns)

            # Update active nodes and connections after mutation
            active_nodes = self._get_active_nodes(node_section)
            active_conns = self._get_active_connections(conn_section)

        # Possibly add a connection
        if (
            self.rng.random() < self.config.conn_add_prob
            and len(active_conns) < self.config.max_connections
        ):
            self._mutate_add_connection(node_section, conn_section, active_nodes)
            active_conns = self._get_active_connections(conn_section)

        # Possibly delete a connection
        if self.rng.random() < self.config.conn_delete_prob and active_conns:
            self._mutate_delete_connection(conn_section, active_conns)

        # Possibly delete a node (only hidden nodes can be deleted)
        hidden_nodes = [n for n in active_nodes if n >= self.config.initial_nodes]
        if self.rng.random() < self.config.node_delete_prob and hidden_nodes:
            self._mutate_delete_node(node_section, conn_section, hidden_nodes)

        # Possibly mutate a node's activation function
        if self.rng.random() < self.config.act_fn_mutate_prod:
            # Focus on output and hidden nodes (typically input/bias activations aren't mutated)
            mutable_nodes = [
                n for n in active_nodes if n >= self.config.output_start_id
            ]

            if mutable_nodes:
                # Pick a random node to mutate
                node_idx = self.rng.choice(mutable_nodes)

                # Change to a different activation function (0-8)
                current_act = int(node_section[node_idx, 2]) % 9
                new_act = (
                    current_act + self.rng.randint(1, 9)
                ) % 9  # Ensures it changes
                node_section[node_idx, 2] = new_act

        # Mutate connection weights
        for idx in active_conns:
            if self.rng.random() < self.config.weight_mutate_prob:
                # Either perturb the weight or assign a new random value
                if self.rng.random() < 0.9:  # 90% chance to perturb
                    conn_section[idx, 2] += self.rng.normal(
                        0, self.config.weight_mutate_power
                    )
                else:
                    conn_section[idx, 2] = self.rng.normal(0, 1.0)  # New random weight

        return new_genome

    def _mutate_add_node(
        self,
        node_section: np.ndarray,
        conn_section: np.ndarray,
        active_conns: List[int],
    ) -> None:
        """Add a node mutation.

        Args:
            node_section: Node section of the genome
            conn_section: Connection section of the genome
            active_conns: List of active connection indices
        """
        # Select a random connection to split
        conn_idx = self.rng.choice(active_conns)
        from_node, to_node, weight, _ = conn_section[conn_idx]

        # Disable the selected connection
        conn_section[conn_idx, 3] = 0

        # Find first free node slot
        for i in range(self.config.initial_nodes, self.config.max_nodes):
            if node_section[i, 0] == 0:  # Check if node is unused
                new_node_id = i
                break
        else:
            return  # No free node slots

        # Set up the new node (as a hidden node)
        node_section[new_node_id] = [
            new_node_id,
            2,
            self.rng.randint(0, 9),
        ]  # random activation

        # Find free connection slots for two new connections
        free_conn_slots = []
        for i in range(self.config.max_connections):
            if conn_section[i, 3] == 0:  # Check if connection is disabled/unused
                free_conn_slots.append(i)
                if len(free_conn_slots) >= 2:
                    break

        if len(free_conn_slots) < 2:
            # Revert changes if not enough connection slots
            conn_section[conn_idx, 3] = 1
            node_section[new_node_id] = [0, 0, 0]
            return

        # Add two new connections
        # 1. Connection from the original source to new node (weight = 1.0)
        conn_section[free_conn_slots[0]] = [from_node, new_node_id, 1.0, 1]

        # 2. Connection from new node to the original target (weight = original weight)
        conn_section[free_conn_slots[1]] = [new_node_id, to_node, weight, 1]

    def _mutate_add_connection(
        self,
        node_section: np.ndarray,
        conn_section: np.ndarray,
        active_nodes: List[int],
    ) -> None:
        """Add a connection mutation.

        Args:
            node_section: Node section of the genome
            conn_section: Connection section of the genome
            active_nodes: List of active node indices
        """
        # Get valid source and target nodes
        # Sources can be input, bias, or hidden nodes
        sources = [
            n for n in active_nodes if node_section[n, 1] in [0, 2]
        ]  # Input, bias, or hidden

        # Targets can be hidden or output nodes
        targets = [
            n for n in active_nodes if node_section[n, 1] in [1, 2]
        ]  # Output or hidden

        if not sources or not targets:
            return

        # Try to find a valid connection that doesn't already exist and won't create a cycle
        max_attempts = 20
        for _ in range(max_attempts):
            from_idx = self.rng.choice(sources)
            to_idx = self.rng.choice(targets)

            # Check if connection already exists
            exists = False
            for i in range(self.config.max_connections):
                if (
                    conn_section[i, 0] == from_idx
                    and conn_section[i, 1] == to_idx
                    and conn_section[i, 3] == 1
                ):
                    exists = True
                    break

            # Skip if connection exists
            if exists:
                continue

            # Check if connection would create a cycle (for feed-forward networks)
            # This is a simple check that ensures from_idx < to_idx
            # A more comprehensive check would use topological sorting
            if from_idx >= to_idx:
                continue

            # Find free connection slot
            for i in range(self.config.max_connections):
                if conn_section[i, 3] == 0:  # Connection is disabled/unused
                    conn_section[i] = [from_idx, to_idx, self.rng.normal(0, 1.0), 1]
                    return

            # No free slots
            return

    def _mutate_delete_connection(
        self, conn_section: np.ndarray, active_conns: List[int]
    ) -> None:
        """Delete a connection mutation.

        Args:
            conn_section: Connection section of the genome
            active_conns: List of active connection indices
        """
        if not active_conns:
            return

        conn_idx = self.rng.choice(active_conns)
        conn_section[conn_idx, 3] = 0  # Disable the connection

    def _mutate_delete_node(
        self,
        node_section: np.ndarray,
        conn_section: np.ndarray,
        hidden_nodes: List[int],
    ) -> None:
        """Delete a node mutation.

        Args:
            node_section: Node section of the genome
            conn_section: Connection section of the genome
            hidden_nodes: List of hidden node indices
        """
        if not hidden_nodes:
            return

        node_idx = self.rng.choice(hidden_nodes)

        # Disable all connections to or from this node
        for i in range(self.config.max_connections):
            if (
                conn_section[i, 0] == node_idx or conn_section[i, 1] == node_idx
            ) and conn_section[i, 3] == 1:
                conn_section[i, 3] = 0

        # Disable the node
        node_section[node_idx] = [0, 0, 0]

    def _get_active_nodes(self, node_section: np.ndarray) -> List[int]:
        """Get indices of active nodes.

        Args:
            node_section: Node section of the genome

        Returns:
            List of active node indices
        """
        return [int(i) for i in range(self.config.max_nodes) if node_section[i, 0] > 0]

    def _get_active_connections(self, conn_section: np.ndarray) -> List[int]:
        """Get indices of active connections.

        Args:
            conn_section: Connection section of the genome

        Returns:
            List of active connection indices
        """
        return [
            i for i in range(self.config.max_connections) if conn_section[i, 3] == 1
        ]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents.

        Args:
            parent1: First parent genome
            parent2: Second parent genome

        Returns:
            Child genome
        """
        # Determine which parent is more fit (always parent1 in this implementation)
        # In a real implementation, you would use fitness values
        more_fit, less_fit = parent1, parent2

        # Create a new genome
        child = np.zeros_like(parent1)

        # Extract node and connection sections
        more_fit_nodes = more_fit[: 3 * self.config.max_nodes].reshape(
            self.config.max_nodes, 3
        )
        less_fit_nodes = less_fit[: 3 * self.config.max_nodes].reshape(
            self.config.max_nodes, 3
        )
        more_fit_conns = more_fit[3 * self.config.max_nodes :].reshape(
            self.config.max_connections, 4
        )
        less_fit_conns = less_fit[3 * self.config.max_nodes :].reshape(
            self.config.max_connections, 4
        )

        child_nodes = child[: 3 * self.config.max_nodes].reshape(
            self.config.max_nodes, 3
        )
        child_conns = child[3 * self.config.max_nodes :].reshape(
            self.config.max_connections, 4
        )

        # Inherit nodes
        for i in range(self.config.max_nodes):
            if more_fit_nodes[i, 0] > 0 and less_fit_nodes[i, 0] > 0:
                # Both parents have this node - randomly choose which to inherit
                if self.rng.random() < 0.5:
                    child_nodes[i] = more_fit_nodes[i]
                else:
                    child_nodes[i] = less_fit_nodes[i]
            elif more_fit_nodes[i, 0] > 0:
                # Only more fit parent has this node - inherit from more fit
                child_nodes[i] = more_fit_nodes[i]

        # Inherit connections
        for i in range(self.config.max_connections):
            more_fit_conn = more_fit_conns[i]
            less_fit_conn = less_fit_conns[i]

            more_fit_enabled = more_fit_conn[3] == 1
            less_fit_enabled = less_fit_conn[3] == 1

            if more_fit_enabled and less_fit_enabled:
                # Both parents have this connection - randomly choose which to inherit
                if self.rng.random() < 0.5:
                    child_conns[i] = more_fit_conn
                else:
                    child_conns[i] = less_fit_conn
            elif more_fit_enabled:
                # Only more fit parent has this connection - inherit from more fit
                child_conns[i] = more_fit_conn

        return child

    def _compute_compatibility_distance(
        self, genome1: np.ndarray, genome2: np.ndarray
    ) -> float:
        """Compute compatibility distance between two genomes.

        Args:
            genome1: First genome
            genome2: Second genome

        Returns:
            Compatibility distance
        """
        # Extract connection sections
        conn1 = genome1[3 * self.config.max_nodes :].reshape(
            self.config.max_connections, 4
        )
        conn2 = genome2[3 * self.config.max_nodes :].reshape(
            self.config.max_connections, 4
        )

        # Get active connections
        active1 = {
            (int(conn1[i, 0]), int(conn1[i, 1])): conn1[i, 2]
            for i in range(self.config.max_connections)
            if conn1[i, 3] == 1
        }
        active2 = {
            (int(conn2[i, 0]), int(conn2[i, 1])): conn2[i, 2]
            for i in range(self.config.max_connections)
            if conn2[i, 3] == 1
        }

        # Count matching and disjoint genes
        matching = 0
        weight_diff = 0.0

        # Count matching genes and calculate weight differences
        for conn in active1:
            if conn in active2:
                matching += 1
                weight_diff += abs(active1[conn] - active2[conn])

        disjoint = len(active1) + len(active2) - 2 * matching

        # Normalize by size of larger genome
        max_size = max(len(active1), len(active2))
        if max_size < 1:
            max_size = 1

        # Compute average weight difference for matching genes
        avg_weight_diff = 0.0
        if matching > 0:
            avg_weight_diff = weight_diff / matching

        # Calculate compatibility distance
        compatibility = (
            self.config.compatibility_disjoint_coefficient * disjoint / max_size
            + self.config.compatibility_weight_coefficient * avg_weight_diff
        )

        return compatibility

    def _speciate(self, genomes: List[np.ndarray]) -> None:
        """Assign genomes to species.

        Args:
            genomes: List of genomes to assign
        """
        # Clear old members but keep representatives
        for species in self.species.values():
            species.members = []

        # Find species for each genome
        for genome in genomes:
            found_species = False

            for species in self.species.values():
                # Check if genome belongs to this species
                if (
                    self._compute_compatibility_distance(genome, species.representative)
                    < self.config.compatibility_threshold
                ):
                    species.members.append(genome)
                    found_species = True
                    break

            if not found_species:
                # Create a new species
                species_id = self.species_counter
                self.species_counter += 1
                self.species[species_id] = Species(species_id, genome)
                self.species[species_id].members.append(genome)

        # Remove empty species
        empty_species = [sid for sid, s in self.species.items() if not s.members]
        for sid in empty_species:
            del self.species[sid]

        # Update representatives
        for species in self.species.values():
            # Choose a random member as the new representative
            if species.members:
                member_idx = (
                    self.rng.randint(0, len(species.members))
                    if len(species.members) > 0
                    else 0
                )
                species.representative = species.members[member_idx].copy()

    def _compute_adjusted_fitness(self, fitnesses: np.ndarray) -> np.ndarray:
        """Compute adjusted fitness values for all genomes.

        Args:
            fitnesses: Raw fitness values

        Returns:
            Adjusted fitness values
        """
        adjusted_fitnesses = np.zeros_like(fitnesses)

        # Map genomes to species
        species_indices = {}
        for i, species in enumerate(self.species.values()):
            for j, genome in enumerate(species.members):
                for k, pop_genome in enumerate(self.population):
                    if np.array_equal(genome, pop_genome):
                        if i not in species_indices:
                            species_indices[i] = []
                        species_indices[i].append(k)

        # Calculate adjusted fitness for each genome
        for i, indices in species_indices.items():
            for idx in indices:
                adjusted_fitnesses[idx] = fitnesses[idx] / len(indices)

        return adjusted_fitnesses

    def ask(self) -> jnp.ndarray:
        """Get current population parameters.

        Returns:
            Population parameters
        """
        return self.jnp_stack(self.population)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        """Update the algorithm based on fitness values.

        Args:
            fitness: Fitness values for each individual
        """
        fitness = np.array(fitness)

        # Track best solution
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self._best_fitness:
            self._best_fitness = fitness[best_idx]
            self._best_params = self.population[best_idx].copy()
            self.logger.info(f"New best fitness: {self._best_fitness}")

        # Speciate the population
        self._speciate(self.population)

        # Compute adjusted fitness (shared fitness within species)
        adjusted_fitness = self._compute_adjusted_fitness(fitness)

        # Allocate offspring to each species based on adjusted fitness
        total_adjusted_fitness = np.sum(adjusted_fitness)

        # Calculate offspring for each species
        species_fitness = {}
        for sid, species in self.species.items():
            species_indices = []
            for i, genome in enumerate(self.population):
                for member in species.members:
                    if np.array_equal(genome, member):
                        species_indices.append(i)
                        break

            if not species_indices:
                species_fitness[sid] = 0
                continue

            species_fitness[sid] = np.sum(adjusted_fitness[species_indices])

        # Calculate number of offspring per species
        species_offspring = {}
        for sid, fit in species_fitness.items():
            if total_adjusted_fitness > 0:
                species_offspring[sid] = max(
                    1, int(fit / total_adjusted_fitness * self.pop_size)
                )
            else:
                species_offspring[sid] = 1

        # Adjust offspring counts to ensure total is pop_size
        total_offspring = sum(species_offspring.values())
        if total_offspring < self.pop_size:
            # Add remaining offspring to highest fitness species
            sorted_species = sorted(
                species_fitness.items(), key=lambda x: x[1], reverse=True
            )
            for sid, _ in sorted_species:
                species_offspring[sid] += 1
                total_offspring += 1
                if total_offspring >= self.pop_size:
                    break
        elif total_offspring > self.pop_size:
            # Remove offspring from lowest fitness species
            sorted_species = sorted(species_fitness.items(), key=lambda x: x[1])
            for sid, _ in sorted_species:
                if species_offspring[sid] > 1:
                    species_offspring[sid] -= 1
                    total_offspring -= 1
                    if total_offspring <= self.pop_size:
                        break

        # Create new population
        new_population = []

        # Elitism - copy best individuals directly
        if self.config.elitism > 0:
            elite_indices = np.argsort(fitness)[-self.config.elitism :]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())

        # Create offspring for each species
        for sid, num_offspring in species_offspring.items():
            species = self.species[sid]
            if not species.members or num_offspring <= 0:
                continue

            # Get fitness for this species
            species_indices = []
            for i, genome in enumerate(self.population):
                for member in species.members:
                    if np.array_equal(genome, member):
                        species_indices.append(i)
                        break

            if not species_indices:
                continue

            species_fitness = fitness[species_indices]

            # Select parents using tournament selection
            species_parents = []
            num_parents = min(
                len(species.members),
                max(1, int(num_offspring * self.config.survival_threshold)),
            )

            # Sort indices by fitness
            sorted_indices = np.argsort(species_fitness)[-num_parents:]

            for idx in sorted_indices:
                species_parents.append(self.population[species_indices[idx]].copy())

            # Create offspring
            remaining = num_offspring - len(new_population)
            if remaining <= 0:
                break

            for _ in range(min(remaining, num_offspring)):
                if len(species_parents) == 1:
                    # Only one parent - just mutate
                    child = self._mutate(species_parents[0].copy())
                else:
                    # Two parents - crossover and mutate
                    parent1 = species_parents[self.rng.randint(len(species_parents))]
                    parent2 = species_parents[self.rng.randint(len(species_parents))]
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)

                new_population.append(child)

        # Fill remaining spots with random offspring
        while len(new_population) < self.pop_size:
            # Select a random genome from the current population
            parent_idx = self.rng.randint(0, len(self.population))
            parent = self.population[parent_idx].copy()
            child = self._mutate(parent.copy())
            new_population.append(child)

        # Update population
        self.population = new_population
        self.generation += 1

        self.logger.info(f"Generation {self.generation} completed")
        self.logger.info(f"Number of species: {len(self.species)}")
        self.logger.info(f"Best fitness: {self._best_fitness}")

    @property
    def best_params(self) -> jnp.ndarray:
        """Get best parameters found so far.

        Returns:
            Best parameters
        """
        return self.jnp_array(self._best_params)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        """Set best parameters.

        Args:
            params: Parameters
        """
        self._best_params = np.array(params)
