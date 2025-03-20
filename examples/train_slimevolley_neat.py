"""Train an agent to solve the SlimeVolley task using NEAT.

Slime Volleyball is a game created in the early 2000s by unknown author.

The game is very simple: the agent's goal is to get the ball to land on
the ground of its opponent's side, causing its opponent to lose a life.

Each agent starts off with five lives. The episode ends when either agent
loses all five lives, or after 3000 timesteps has passed. An agent receives
a reward of +1 when its opponent loses or -1 when it loses a life.

An agent loses when it loses 5 times in the Test environment, or if it
loses based on score count after 3000 time steps.

During Training, the game is simply played for 3000 time steps, not
terminating even when one player loses 5 times.

This task is based on:
https://otoro.net/slimevolley/
https://github.com/hardmaru/slimevolleygym

Example command to run this script: `python train_slimevolley_neat.py --gpu-id=0`
"""

import argparse
import os
import shutil
import jax

from evojax.task.slimevolley import SlimeVolley
from evojax.policy.neat_policy import NEATPolicy  # Import the NEAT policy
from evojax.algo.neat_algo import NEAT  # Import the NEAT solver
from evojax import Trainer
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()

    # Basic training parameters
    parser.add_argument(
        "--pop-size", type=int, default=30, help="NEAT population size."
    )
    parser.add_argument(
        "--max-iter", type=int, default=500, help="Max training iterations."
    )
    parser.add_argument(
        "--n-repeats", type=int, default=8, help="Training repetitions."
    )
    parser.add_argument(
        "--num-tests", type=int, default=30, help="Number of test rollouts."
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for training."
    )

    # Network structure parameters
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=30,
        help="Maximum number of nodes in NEAT network.",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=100,
        help="Maximum number of connections in NEAT network.",
    )
    parser.add_argument("--init-std", type=float, default=0.5, help="Initial std.")

    # NEAT specific parameters (mutation rates)
    parser.add_argument(
        "--conn-add-prob",
        type=float,
        default=0.15,
        help="Probability of adding a connection",
    )
    parser.add_argument(
        "--conn-delete-prob",
        type=float,
        default=0.08,
        help="Probability of deleting a connection",
    )
    parser.add_argument(
        "--node-add-prob", type=float, default=0.1, help="Probability of adding a node"
    )
    parser.add_argument(
        "--node-delete-prob",
        type=float,
        default=0.03,
        help="Probability of deleting a node",
    )
    parser.add_argument(
        "--weight-mutate-prob",
        type=float,
        default=0.9,
        help="Probability of mutating weights",
    )
    parser.add_argument(
        "--weight-mutate-power",
        type=float,
        default=2.0,
        help="Power/magnitude of weight mutations",
    )
    parser.add_argument(
        "--act-fn-mutate-prod",
        type=float,
        default=0.1,
        help="Probability of mutating activation functions",
    )

    # Speciation parameters
    parser.add_argument(
        "--compatibility-threshold",
        type=float,
        default=3.8,
        help="Threshold for speciation",
    )
    parser.add_argument(
        "--compatibility-disjoint-coef",
        type=float,
        default=1.0,
        help="Disjoint genes coefficient for compatibility",
    )
    parser.add_argument(
        "--compatibility-weight-coef",
        type=float,
        default=0.6,
        help="Weight coefficient for compatibility",
    )
    parser.add_argument(
        "--survival-threshold",
        type=float,
        default=0.3,
        help="Fraction of each species to keep for reproduction",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=3,
        help="Number of top individuals to preserve unchanged",
    )

    # Logging and debugging
    parser.add_argument("--test-interval", type=int, default=20, help="Test interval.")
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Logging interval."
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--gpu-id", type=str, help="GPU(s) to use.")

    config, _ = parser.parse_known_args()
    return config


def visualize_neat_network(
    params, max_nodes, max_connections, input_dim, output_dim, output_file
):
    """Visualize the NEAT network structure without graphviz dependency.

    Args:
        params: Network parameters (genome)
        max_nodes: Maximum number of nodes
        max_connections: Maximum number of connections
        input_dim: Number of input nodes
        output_dim: Number of output nodes
        output_file: Output file path
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract node and connection sections
        node_section = params[: 3 * max_nodes].reshape(max_nodes, 3)
        conn_section = params[3 * max_nodes :].reshape(max_connections, 4)

        # Identify active nodes and connections
        active_nodes = set()

        # Add input nodes
        for i in range(input_dim):
            active_nodes.add(i)

        # Add bias node
        bias_node = input_dim
        active_nodes.add(bias_node)

        # Add output nodes
        for i in range(output_dim):
            node_id = input_dim + 1 + i
            active_nodes.add(node_id)

        # Add hidden nodes
        for i in range(input_dim + 1 + output_dim, max_nodes):
            if node_section[i, 0] > 0:  # Active node
                active_nodes.add(i)

        # Collect enabled connections
        connections = []
        for i in range(max_connections):
            if conn_section[i, 3] == 1:  # Enabled connection
                from_node = int(conn_section[i, 0])
                to_node = int(conn_section[i, 1])
                weight = conn_section[i, 2]

                if from_node in active_nodes and to_node in active_nodes:
                    connections.append((from_node, to_node, weight))

        # Create a better node layout manually
        # Organize nodes in layers
        input_layer = list(range(input_dim))
        bias_layer = [bias_node]
        output_layer = list(range(input_dim + 1, input_dim + 1 + output_dim))

        # Find hidden nodes and organize by connectivity
        hidden_nodes = [
            n for n in active_nodes if n not in input_layer + bias_layer + output_layer
        ]

        # Determine how many hidden layers to use
        num_hidden_layers = min(3, len(hidden_nodes))
        if num_hidden_layers == 0 and hidden_nodes:
            num_hidden_layers = 1

        # Assign hidden nodes to layers
        hidden_layers = [[] for _ in range(num_hidden_layers)]

        if hidden_nodes:
            # Simple distribution of hidden nodes across layers
            for i, node in enumerate(hidden_nodes):
                layer_idx = i % num_hidden_layers
                hidden_layers[layer_idx].append(node)

        # Create positions dictionary
        pos = {}

        # Position input layer
        input_y_spacing = 1.0 if input_dim > 1 else 0.5
        for i, node in enumerate(input_layer):
            pos[node] = (0, (i - (input_dim - 1) / 2) * input_y_spacing)

        # Position bias node above the input layer
        pos[bias_node] = (0, ((input_dim - 1) / 2 + 1) * input_y_spacing)

        # Position output layer
        output_y_spacing = 1.0 if output_dim > 1 else 0.5
        for i, node in enumerate(output_layer):
            pos[node] = (
                num_hidden_layers + 2,
                (i - (output_dim - 1) / 2) * output_y_spacing,
            )

        # Position hidden layers
        for layer_idx, layer_nodes in enumerate(hidden_layers):
            if not layer_nodes:
                continue

            y_spacing = 1.0 if len(layer_nodes) > 1 else 0.5
            for i, node in enumerate(layer_nodes):
                pos[node] = (
                    layer_idx + 1,
                    (i - (len(layer_nodes) - 1) / 2) * y_spacing,
                )

        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # Draw connections
        max_weight = max([abs(w) for _, _, w in connections]) if connections else 1.0

        for from_node, to_node, weight in connections:
            if from_node not in pos or to_node not in pos:
                continue

            start_x, start_y = pos[from_node]
            end_x, end_y = pos[to_node]

            # Normalize line width
            line_width = 1.0 + 3.0 * abs(weight) / max_weight

            # Choose color based on weight sign
            color = "green" if weight > 0 else "red"

            # Draw the curve
            ax.annotate(
                "",
                xy=(end_x, end_y),
                xycoords="data",
                xytext=(start_x, start_y),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle=f"arc3,rad={0.1 * np.sign(weight) * np.random.random()}",
                    color=color,
                    lw=line_width,
                    alpha=0.7,
                ),
            )

            # Add weight text for important connections
            if abs(weight) > 0.5 * max_weight:
                text_x = (start_x + end_x) / 2 + 0.1
                text_y = (start_y + end_y) / 2 + 0.1
                plt.text(
                    text_x,
                    text_y,
                    f"{weight:.2f}",
                    fontsize=8,
                    color=color,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

        # Draw nodes
        node_sizes = []
        node_colors = []
        for node in pos:
            if node in input_layer:
                node_colors.append("skyblue")
                node_sizes.append(600)
            elif node == bias_node:
                node_colors.append("gold")
                node_sizes.append(600)
            elif node in output_layer:
                node_colors.append("lightgreen")
                node_sizes.append(600)
            else:
                # For hidden nodes, use activation function color
                try:
                    # Get the activation function type from node params
                    node_idx = list(active_nodes).index(node)
                    activation = node_section[node, 0]
                    if activation > 0.66:
                        node_colors.append("lightcoral")  # tanh
                    elif activation > 0.33:
                        node_colors.append("plum")  # sigmoid
                    else:
                        node_colors.append("lightgray")  # relu
                except:
                    node_colors.append("lightgray")
                node_sizes.append(500)

        # Extract x and y coordinates
        xs = [pos[node][0] for node in pos]
        ys = [pos[node][1] for node in pos]

        # Draw the nodes
        plt.scatter(xs, ys, s=node_sizes, c=node_colors, edgecolors="black", zorder=10)

        # Add node labels
        for node in pos:
            x, y = pos[node]
            if node in input_layer:
                label = f"In {node}"
            elif node == bias_node:
                label = "Bias"
            elif node in output_layer:
                label = f"Out {node - (input_dim + 1)}"
            else:
                label = f"H {node}"

            plt.text(
                x,
                y,
                label,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Add title and info
        plt.title(
            f"NEAT Network Structure ({len(active_nodes)} nodes, {len(connections)} connections)"
        )

        # Remove axes
        plt.axis("off")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight", dpi=150)
        plt.close()

        print(f"Network visualization saved to {output_file}")

    except ImportError:
        print("Could not visualize network. Please install matplotlib.")
    except Exception as e:
        print(f"Error visualizing network: {e}")
        import traceback

        traceback.print_exc()


def visualize_policy(test_task, policy, params, max_steps, gif_file):
    """Visualize the policy in action.

    Args:
        test_task: SlimeVolley test task
        policy: NEAT policy
        params: Network parameters
        max_steps: Maximum number of steps to visualize
        gif_file: Output GIF file path
    """
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)

    # Add batch dimension
    batched_params = params[None, :]
    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)
    screens = []

    print("Running simulation for visualization...")
    for step in range(max_steps):
        action, policy_state = action_fn(task_state, batched_params, policy_state)
        task_state, _, _ = step_fn(task_state, action)
        screens.append(SlimeVolley.render(task_state))

    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0
    )
    print(f"GIF saved to {gif_file}")


def main(config):
    log_dir = "./log/slimevolley_neat"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name="SlimeVolley_NEAT", log_dir=log_dir, debug=config.debug
    )
    logger.info("EvoJAX SlimeVolley with NEAT")
    logger.info("=" * 30)

    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)

    # Create NEAT policy
    policy = NEATPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        output_act_fn="tanh",
        max_nodes=config.max_nodes,
        max_connections=config.max_connections,
        logger=logger,
    )

    # Use NEAT solver
    solver = NEAT(
        param_size=policy.num_params,
        pop_size=config.pop_size,
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        max_nodes=config.max_nodes,
        max_connections=config.max_connections,
        init_stdev=config.init_std,
        seed=config.seed,
        logger=logger,
        compatibility_threshold=config.compatibility_threshold,
        compatibility_disjoint_coefficient=config.compatibility_disjoint_coef,
        compatibility_weight_coefficient=config.compatibility_weight_coef,
        conn_add_prob=config.conn_add_prob,
        conn_delete_prob=config.conn_delete_prob,
        node_add_prob=config.node_add_prob,
        node_delete_prob=config.node_delete_prob,
        act_fn_mutate_prod=config.act_fn_mutate_prod,
        weight_mutate_prob=config.weight_mutate_prob,
        weight_mutate_power=config.weight_mutate_power,
        survival_threshold=config.survival_threshold,
        elitism=config.elitism,
    )

    # Train
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model
    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Visualize the evolved network structure
    visualize_neat_network(
        trainer.solver.best_params,
        config.max_nodes,
        config.max_connections,
        train_task.obs_shape[0],
        train_task.act_shape[0],
        os.path.join(log_dir, f"network_structure_neat.png"),
    )

    # Visualize the policy with best parameters
    logger.info("Generating GIF with best parameters from training...")
    visualize_policy(
        test_task,
        policy,
        trainer.solver.best_params,
        max_steps,
        os.path.join(log_dir, f"slimevolley_neat.gif"),
    )


if __name__ == "__main__":
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_id
    main(configs)
