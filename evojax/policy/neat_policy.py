"""Implementation of the NEAT policy network.

This module provides a policy network implementation based on the NEAT
(NeuroEvolution of Augmenting Topologies) algorithm for use with EvoJAX.

The NEATPolicy class represents neural networks with evolving topologies where:
- Networks start with minimal structure (direct input-output connections)
- Both structure (nodes and connections) and weights evolve over time
- Different nodes can use different activation functions within the same network
- Only feed-forward architectures are supported (no recurrent connections)
- Networks are represented with fixed-size arrays for JAX compatibility

The implementation supports multiple activation functions:
- Hyperbolic tangent (tanh)
- Sigmoid
- SELU (Scaled Exponential Linear Unit)
- LeakyReLU
- ELU (Exponential Linear Unit)
- Swish
- Identity (linear)
- GELU (Gaussian Error Linear Unit)
- Softplus

This class works in conjunction with the NEAT algorithm to evolve network
topologies that solve reinforcement learning tasks efficiently.
"""

import logging
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger


class NEATPolicy(PolicyNetwork):
    """A policy network using NEAT (NeuroEvolution of Augmenting Topologies).

    This implementation supports feed-forward networks only (no recurrent connections).
    It allows different activation functions for different nodes within the same network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_act_fn: str = "tanh",
        max_nodes: int = 100,
        max_connections: int = 500,
        max_propagation_steps: int = 5,
        logger: logging.Logger = None,
    ):
        """Initialize a NEAT policy.

        Args:
            input_dim: Dimensionality of the input space.
            output_dim: Dimensionality of the output space.
            output_act_fn: Activation function for output layer.
            max_nodes: Maximum number of nodes in the network.
            max_connections: Maximum number of connections in the network.
            max_propagation_steps: Maximum iterations for signal propagation.
            logger: Logger instance.
        """
        if logger is None:
            self._logger = create_logger(name="NEATPolicy")
        else:
            self._logger = logger

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._output_act_fn = output_act_fn
        self._max_nodes = max_nodes
        self._max_connections = max_connections
        self._max_propagation_steps = max_propagation_steps

        # Initial number of nodes: inputs + outputs + bias
        self._initial_nodes = input_dim + output_dim + 1
        self._bias_idx = input_dim
        self._output_start_idx = input_dim + 1
        self._output_end_idx = self._output_start_idx + output_dim

        # Calculate parameter size for genome encoding
        # Each connection needs: from_node, to_node, weight, enabled flag
        # Each node needs: node_id, node_type, activation
        # We use a fixed-size representation for JAX compatibility
        connection_params = 4 * max_connections  # from, to, weight, enabled
        node_params = 3 * max_nodes  # id, type, activation
        self.num_params = connection_params + node_params

        self._logger.info(f"NEATPolicy.num_params = {self.num_params}")

        # Available activation functions
        self._activation_fns = [
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
        self._num_activations = len(self._activation_fns)

        # Name mapping for logging and visualization
        self._activation_names = [
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

        # Setup the forward pass function
        self._forward_fn = jax.vmap(self._forward)

    def _format_genome(self, flat_params: jnp.ndarray) -> Dict:
        """Convert flat parameter array to structured genome.

        Args:
            flat_params: Flat array of parameters.

        Returns:
            Dictionary containing structured genome.
        """
        # Split params into node and connection sections
        node_params = flat_params[: 3 * self._max_nodes]
        conn_params = flat_params[3 * self._max_nodes :]

        # Reshape node params: [id, type, activation]
        node_params = node_params.reshape(self._max_nodes, 3)

        # Reshape connection params: [from_node, to_node, weight, enabled]
        conn_params = conn_params.reshape(self._max_connections, 4)

        return {"nodes": node_params, "connections": conn_params}

    def _apply_activation(self, value, act_type):
        """Apply the appropriate activation function based on activation type.

        Args:
            value: The input value to the activation function.
            act_type: Index of the activation function to use.

        Returns:
            The activated value.
        """
        # Convert activation type to an index by taking modulo of activation types
        # This ensures even if a mutation creates an invalid index, it maps to a valid one
        act_idx = jnp.int32(jnp.abs(act_type) % self._num_activations)

        # Initialize with first activation
        result = self._activation_fns[0](value)

        # Use lax.select to choose the correct activation (JAX-friendly conditional)
        for i in range(1, self._num_activations):
            result = lax.select(act_idx == i, self._activation_fns[i](value), result)

        return result

    def _propagate_step(
        self, node_values: jnp.ndarray, nodes: jnp.ndarray, connections: jnp.ndarray
    ) -> jnp.ndarray:
        """Execute one step of forward propagation through the network.

        Args:
            node_values: Current node values.
            nodes: Node parameters.
            connections: Connection parameters.

        Returns:
            Updated node values.
        """
        # Create a fresh node activation buffer to collect incoming signals
        node_activations = jnp.zeros(self._max_nodes)

        # Filter for enabled connections (value of 1.0 in the enabled flag column)
        enabled_mask = connections[:, 3] > 0.5  # Enabled flag must be > 0.5

        # Only process enabled connections
        def process_connection(i, activations):
            # Skip if not enabled
            enabled = connections[i, 3] > 0.5

            # Extract connection data
            from_node = jnp.int32(connections[i, 0])
            to_node = jnp.int32(connections[i, 1])
            weight = connections[i, 2]

            # Get source value
            from_value = node_values[from_node]

            # Add weighted input to the target node's activation
            # Only update if the connection is enabled
            return lax.cond(
                enabled,
                lambda: activations.at[to_node].add(from_value * weight),
                lambda: activations,
            )

        # Process all connections
        for i in range(self._max_connections):
            node_activations = process_connection(i, node_activations)

        # Update hidden and output nodes by applying activation functions
        # Start from the first output node (bias and inputs remain unchanged)
        def update_node_value(i, values):
            # Get activation for this node
            act_type = nodes[i, 0]

            # Get accumulated activation
            activation = node_activations[i]

            # Apply the appropriate activation function
            activated_value = self._apply_activation(activation, act_type)

            # Update the node value
            return values.at[i].set(activated_value)

        # Update output nodes
        for i in range(self._output_start_idx, self._output_end_idx):
            node_values = update_node_value(i, node_values)

        # Update hidden nodes (everything after outputs)
        for i in range(self._output_end_idx, self._max_nodes):
            # Only update if node is active - node considered active if any parameter is non-zero
            is_active = jnp.any(jnp.abs(nodes[i]) > 1e-6)
            node_values = lax.cond(
                is_active,
                lambda: update_node_value(i, node_values),
                lambda: node_values,
            )

        return node_values

    def _forward(self, params: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for a single NEAT network.

        Args:
            params: Flat array of parameters representing the genome.
            obs: Observation from the environment.

        Returns:
            Action vector.
        """
        genome = self._format_genome(params)
        nodes = genome["nodes"]
        connections = genome["connections"]

        # Initialize node values (including bias node set to 1.0)
        node_values = jnp.zeros(self._max_nodes)

        # Set input values
        # Input nodes are positioned at the start (0 to input_dim-1)
        node_values = node_values.at[: self._input_dim].set(obs)

        # Set bias node (positioned after input nodes)
        node_values = node_values.at[self._bias_idx].set(1.0)

        # Forward propagation loop - fixed number of iterations for JAX compatibility
        for _ in range(self._max_propagation_steps):
            node_values = self._propagate_step(node_values, nodes, connections)

        # Get output values
        output_values = node_values[self._output_start_idx : self._output_end_idx]

        # No need to apply activation again - already applied in propagate_step
        return output_values

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        """Get actions for batched observations.

        Args:
            t_states: Task states containing observations.
            params: Parameters (genomes) for each policy in the batch.
            p_states: Policy states (unused in this implementation).

        Returns:
            Tuple of (actions, new_policy_states).
        """
        return self._forward_fn(params, t_states.obs), p_states

    def reset(self, t_states: TaskState) -> PolicyState:
        """Reset the policy state.

        Args:
            t_states: Task states

        Returns:
            Initial policy state
        """
        # For this stateless policy, we simply return None
        return None

    def get_activation_name(self, act_idx):
        """Get the name of an activation function for visualization purposes.

        Args:
            act_idx: Index of the activation function.

        Returns:
            String name of the activation function.
        """
        # Apply same modulo logic as in _apply_activation
        idx = abs(int(act_idx)) % self._num_activations
        return self._activation_names[idx]
