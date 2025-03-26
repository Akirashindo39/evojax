"""Constants for NEAT algorithm.

This module provides default constants and configuration values for the
NEAT (NeuroEvolution of Augmenting Topologies) algorithm. Centralizing
these values makes it easier to maintain consistent configurations and
reduces hard coded numbers in the code.
"""

import jax
import jax.numpy as jnp

# Network structure parameters
DEFAULT_MAX_NODES = 30
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_PROPAGATION_STEPS = 5

# Speciation parameters
DEFAULT_COMPATIBILITY_THRESHOLD = 3.8
DEFAULT_COMPATIBILITY_DISJOINT_COEFFICIENT = 1.0
DEFAULT_COMPATIBILITY_WEIGHT_COEFFICIENT = 0.6

# Structure mutation parameters
DEFAULT_CONN_ADD_PROB = 0.15
DEFAULT_CONN_DELETE_PROB = 0.08
DEFAULT_NODE_ADD_PROB = 0.1
DEFAULT_NODE_DELETE_PROB = 0.03
DEFAULT_ACT_FN_MUTATE_PROB = 0.1

# Weight mutation parameters
DEFAULT_WEIGHT_MUTATE_PROB = 0.9
DEFAULT_WEIGHT_MUTATE_POWER = 2.0

# Selection parameters
DEFAULT_SURVIVAL_THRESHOLD = 0.3
DEFAULT_ELITISM = 3

# Training parameters
DEFAULT_POPULATION_SIZE = 30
DEFAULT_MAX_ITERATIONS = 500
DEFAULT_NUM_REPEATS = 8
DEFAULT_NUM_TESTS = 30
DEFAULT_INIT_STD = 0.5
DEFAULT_SEED = 123
DEFAULT_TEST_INTERVAL = 20
DEFAULT_LOG_INTERVAL = 10

# Node types
NODE_TYPE_INPUT = 0
NODE_TYPE_OUTPUT = 1
NODE_TYPE_HIDDEN = 2

# Activation function indices
ACT_TANH = 0
ACT_SIGMOID = 1
ACT_SELU = 2
ACT_LEAKY_RELU = 3
ACT_ELU = 4
ACT_SWISH = 5
ACT_IDENTITY = 6
ACT_GELU = 7
ACT_SOFTPLUS = 8
NUM_ACT_FNS = 9  # Total number of available activation functions

# Activation function names (for visualization)
ACT_NAMES = [
    "tanh",
    "sigmoid",
    "selu",
    "leaky_relu",
    "elu",
    "swish",
    "identity",
    "gelu",
    "softplus",
]

# Available activation functions
ACT_FNS = [
    jnp.tanh,  # 0: tanh - maps to [-1, 1]
    jax.nn.sigmoid,  # 1: sigmoid - maps to [0, 1]
    jax.nn.selu,  # 2: SELU - has self-normalizing properties
    lambda x: jnp.maximum(0.01 * x, x),  # 3: LeakyReLU - differentiable at 0
    jax.nn.elu,  # 4: ELU - differentiable alternative to ReLU
    jax.nn.swish,  # 5: Swish - smooth activation function
    lambda x: x,  # 6: Identity - linear activation
    jax.nn.gelu,  # 7: GELU - smooth, performant in deep networks
    jax.nn.softplus,  # 8: Softplus - smooth approximation of ReLU
]

# Connection parameters
CONNECTION_ENABLED = 1
CONNECTION_DISABLED = 0

# SlimeVolley specific constants
SLIMEVOLLEY_MAX_STEPS = 3000
