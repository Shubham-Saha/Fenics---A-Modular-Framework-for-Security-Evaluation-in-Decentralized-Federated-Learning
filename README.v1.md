# Fenics: Decentralized Federated Learning Simulator

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-purple.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0%2B-yellow.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-2.5%2B-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-red.svg)
![Threading](https://img.shields.io/badge/Concurrent-Multithreading-blueviolet.svg)
![Modular](https://img.shields.io/badge/Architecture-Modular-brightgreen.svg)

**Fenics** is a comprehensive simulator designed for decentralized federated learning environments. It allows researchers and practitioners to experiment with various network topologies, participant selection strategies, and attack scenarios to evaluate the robustness and efficiency of federated learning algorithms.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Installing via `pip`](#installing-via-pip)
  - [Uninstallation](#uninstallation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
  - [Parameters in `config.yaml`](#parameters-in-configyaml)
  - [Setting Up `topology.edgelist`](#setting-up-topologyedgelist)
- [Usage](#usage)
  - [Using the Fenics Shell](#using-the-fenics-shell)
  - [Running with Command-Line Arguments](#running-with-command-line-arguments)
- [Extending Fenics](#extending-fenics)
  - [Adding Custom Topologies](#adding-custom-topologies)
  - [Adding Custom Models](#adding-custom-models)
  - [Adding Custom Client Selection Strategies](#adding-custom-client-selection-strategies)
  - [Adding Custom Attack Types](#adding-custom-attack-types)
  - [Adding Custom Aggregation Strategies](#adding-custom-aggregation-strategies)
  - [Adding Custom Communication Protocols](#adding-custom-communication-protocols)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Customizable Network Topologies:** Define and utilize custom network structures via edge list files or built-in topologies
- **Flexible Participant Selection:** Implement various client selection strategies based on data size or other metrics
- **Support for Attacks:** Simulate adversarial attacks to assess the resilience of federated learning models
- **Modular Architecture:** Extensible components for models, topologies, attack types, and more
- **Comprehensive Logging and Visualization:** Generate detailed logs and visual plots to analyze simulation results
- **Scalable Simulations:** Handle simulations with varying numbers of nodes and complex network structures
- **Real-Time Progress Monitoring:** Visualize simulation progress with an integrated loading bar

---

## Prerequisites

Before installing and using Fenics, ensure that your system meets the following requirements:

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.7 or higher
- **Virtual Environment (Recommended):** It's advisable to use a virtual environment to manage dependencies

---

## Installation

### Installing via `pip`

1. **Clone the Repository:**
    
    ```bash
    git clone https://github.com/yourusername/fenics.git
    cd fenics
    ```
    
2. **Create and Activate a Virtual Environment (Optional but Recommended):**
    
    - **Windows:**
    
      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```
    
    - **Unix/Linux:**
    
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    
3. **Install Required Dependencies:**
    
    Ensure you have `pip` updated to the latest version:
    
    ```bash
    pip install --upgrade pip
    ```
    
4. **Install the Package in Editable Mode:**
    
    This allows you to modify the code without reinstalling the package each time.
    
    ```bash
    pip install -e .
    ```
    
    **Note:** The `setup.py` is configured to install all necessary dependencies, including `torch`, `numpy`, `psutil`, `colorama`, `pyfiglet`, and `pydantic`.
    
5. **Verify Installation:**
    
    After installation, the `fenics` command-line tool should be available.
    
    ```bash
    fenics --help
    ```

    **Expected Output:**
    
    ```
    Usage: phoenix [OPTIONS] COMMAND [ARGS]...
    
      Distributed Federated Learning Simulator
    
    Options:
      --help  Show this message and exit.
    
    Commands:
      setup            Initialize the simulation environment with desired...
      run              Execute the simulation with the provided options.
      list_simulations List all available simulation configurations.
      parameters       Display all available simulation parameters and...
    ```

### Uninstallation

If you need to uninstall Fenics, follow these steps:
    
1. **Deactivate the Virtual Environment (If Active):**
    
    ```bash
    deactivate
    ```
    
2. **Uninstall the Package:**
    
    ```bash
    pip uninstall fenics
    ```
    
3. **Remove the Virtual Environment (Optional):**
    
    If you created a virtual environment specifically for Fenics, you can delete it:
    
    - **Windows:**
    
      ```bash
      rmdir /s /q venv
      ```
    
    - **Unix/Linux:**
    
      ```bash
      rm -rf venv
      ```

---

## Project Structure

Fenics follows a modular architecture with the following components:

```
fenics/
├── fenics/
│   ├── __init__.py
│   ├── aggregation/          # Aggregation strategies (FedAvg, etc.)
│   ├── attack/               # Attack implementations (poison, delay)
│   ├── cli/                  # CLI interface components
│   ├── client_selection/     # Client selection strategies
│   ├── communication/        # Communication protocols
│   ├── data/                 # Data handling and loading
│   ├── models/               # ML model implementations
│   ├── plotting/             # Visualization utilities
│   ├── topology/             # Network topology implementations
│   ├── training/             # Training and evaluation
│   ├── config.py             # Configuration handling
│   ├── simulator.py          # Main simulation logic
│   └── utils.py              # Utility functions
├── config.yaml               # Configuration file
├── fenics.py                 # Main entry point
├── setup.py                  # Package setup
├── LICENSE
└── README.md
```

Each module is designed to be extensible, allowing you to add custom implementations without modifying the core code.

---

## Configuration

Fenics uses a `config.yaml` file to manage simulation parameters. Proper configuration ensures that simulations run as intended with the desired settings.
    
### Parameters in `config.yaml`
    
Below is an explanation of the primary parameters you can configure:
    
```yaml
simulations:
  simulation1:
    rounds: 3                   # Number of training rounds
    epochs: 1                   # Number of epochs per round
    num_nodes: 5                # Total number of nodes in the simulation
    num_attackers: 0            # Number of attacker nodes
    attacker_nodes: []          # Specific nodes designated as attackers
    attacks: []                 # Types of attacks to simulate
    use_attackers: false        # Flag to enable or disable attacker simulation
    participation_rate: 0.6     # Fraction of nodes participating each round
    topology: fully_connected   # Network topology type
    topology_file: null         # Path to a custom topology edge list file
    max_attacks: null           # Maximum number of attacks an attacker can perform
    gossip_steps: 3             # Number of gossiping steps during aggregation
    protocol: gossip            # Aggregation protocol ('gossip' or 'neighboring')
    alpha: 0.5                  # Dirichlet distribution parameter for data partitioning
    model_type: cnn             # Model type to use ('cnn', 'mlp', or custom)
```
    
**Parameter Descriptions:**
    
- **`rounds`**: Total number of training rounds to execute.
- **`epochs`**: Number of training epochs each node performs per round.
- **`num_nodes`**: Total number of participating nodes in the simulation.
- **`num_attackers`**: Number of nodes designated as attackers.
- **`attacker_nodes`**: List of specific node IDs to act as attackers. Overrides `num_attackers` if provided.
- **`attacks`**: Types of attacks to simulate (e.g., 'delay', 'poison'). Define the attack strategies you want to test.
- **`use_attackers`**: Boolean flag to enable (`true`) or disable (`false`) attacker simulations.
- **`participation_rate`**: Fraction of nodes selected to participate in each round. For example, `0.6` means 60% of nodes are selected each round.
- **`topology`**: Defines the network topology. Common options include:
  - `fully_connected`: Every node is connected to every other node.
  - `ring`: Nodes are connected in a ring structure.
  - `random`: Random graph with probabilistic connections.
  - `custom`: Use a custom topology defined in an edge list file.
- **`topology_file`**: Path to a custom topology edge list file. Required if `topology` is set to `custom`.
- **`max_attacks`**: Maximum number of attacks an attacker node can perform throughout the simulation.
- **`gossip_steps`**: Number of gossiping steps to perform during the aggregation phase.
- **`protocol`**: Communication protocol to use. Options:
  - `gossip`: Nodes exchange updates in a gossip manner.
  - `neighboring`: Nodes exchange updates with their immediate neighbors.
- **`alpha`**: Parameter for the Dirichlet distribution used in data partitioning among nodes.
- **`model_type`**: Type of model to use. Options:
  - `cnn`: Convolutional Neural Network (default).
  - `mlp`: Multi-Layer Perceptron.
  - Custom models can be registered and used.
    
### Setting Up `topology.edgelist`
    
When using a **custom network topology**, you need to define the connections between nodes in an edge list file. Here's how to set it up:
    
1. **Create the Edge List File:**
    
    - **Filename:** `topology.edgelist` (or any name of your choice)
    - **Format:** Each line represents an undirected edge between two nodes, specified by their node IDs separated by a space.
    
    **Example `topology.edgelist`:**
    
    ```
    0 1
    1 2
    1 4
    2 3
    3 5
    3 6
    4 5
    5 6
    5 7
    7 8
    8 9
    ```
    
    **Explanation:**
    - Node `0` is connected to Node `1`.
    - Node `1` is connected to Nodes `0`, `2`, and `4`.
    - And so on...
    
2. **Place the Edge List File:**
    
    Save the `topology.edgelist` file in a directory accessible to your project. For example, you can place it in the root directory of your project.
    
3. **Configure `config.yaml` to Use the Custom Topology:**
    
    Update the `config.yaml` to reference your custom edge list file.
    
    ```yaml
    simulations:
      simulation1:
        # Other parameters...
        topology: custom
        topology_file: path/to/your/topology.edgelist
    ```
4. **Understanding the `topology.edgelist`:**
    
    - **Node IDs:** Ensure that node IDs in the edge list start from `0` and are consecutive integers up to `num_nodes - 1`.
    - **Undirected Edges:** Each connection is bidirectional. If you want Node `0` connected to Node `1`, you only need to specify `0 1`. There's no need to add `1 0`.
    
---

## Usage
    
Fenics provides a command-line interface called **Fenics Shell** to interact with the simulator. Below are instructions on how to set up and run simulations.
    
### Using the Fenics Shell
    
1. **Launch the Fenics Shell:**
    
    ```bash
    fenics
    ```
    
    **Sample Output:**
    
    ```
       ______              _            
      |  ____|            (_)           
      | |__ ___ _ __  ___ _  ___ ___ 
      |  __/ _ \ '_ \/ __| |/ __/ __|
      | | |  __/ | | \__ \ | (__\__ \
      |_|  \___|_| |_|___/_|\___|___/
                                        
    Welcome to Fenics Shell! Type 'help' to see available commands.
    Fenics> 
    ```
    
2. **Setup the Simulation Environment:**
    
    Use the `setup` command to initialize simulation parameters.
    
    - **Using Direct Parameters:**
    
      ```bash
      setup --rounds 3 --epochs 1 --topology fully_connected --participation_rate 0.6 --protocol gossip --num_nodes 5 --alpha 0.5
      ```
    
    - **Using a Configuration File:**
    
      ```bash
      setup --config config.yaml --simulation_name simulation1
      ```
    
      **Explanation:**
      - **`--config`**: Specifies the path to the `config.yaml` file.
      - **`--simulation_name`**: Identifies which simulation parameters to load from the configuration file.
    
3. **Run the Simulation:**
    
    After setup, execute the simulation using the `run` command.
    
    ```bash
    run
    ```
    
    **Sample Output:**
    
    ```
    Final Simulation Arguments:
    Rounds: 3
    Epochs: 1
    Topology: fully_connected
    Participation rate: 0.6
    Protocol: gossip
    Num nodes: 5
    Alpha: 0.5
    
    Starting simulation...
    Simulation Progress: 100%|██████████████████████████████████████████████████████████| 3/3 [00:01<00:00, 2.50round/s, CPU Usage: 55%]
    
    Simulation completed successfully. Check the 'results' directory for outputs.
    ```
    
    **Explanation:**
    - **Progress Bar:** A real-time progress bar displays the simulation's progress, showing the percentage completed and the latest CPU usage.
    - **Logs and Plots:** Detailed logs and visual plots are saved in the `results` directory.
    
4. **Exit the Fenics Shell:**
    
    Use the `exit` or `quit` command to exit.
    
    ```bash
    exit
    ```
    
### Running with Command-Line Arguments
    
Alternatively, you can run simulations directly without entering the Fenics Shell by passing commands as arguments.

```bash
fenics setup --config config.yaml --simulation_name simulation1 && fenics run
```
    
**Note:** This approach sequentially executes the `setup` and `run` commands.

---

## Extending Fenics

Fenics is designed to be modular and extensible. You can easily add custom implementations for various components without modifying the core code.

### Adding Custom Topologies

1. Create a new Python file in the `fenics/topology/` directory:

```python
# fenics/topology/my_topology.py
from fenics.topology import TopologyBase, TopologyFactory
import networkx as nx

class MyTopology(TopologyBase):
    def __init__(self, num_nodes, **kwargs):
        super().__init__(num_nodes)
        # Additional initialization if needed
        
    def build(self):
        # Implement your topology building logic
        G = nx.Graph()
        # Add nodes and edges
        return G

# Register the topology
TopologyFactory.register_topology('my_topology', MyTopology)
```

2. Import your file and use it:

```python
import fenics.topology.my_topology

# Use in configuration
# topology: my_topology
```

### Adding Custom Models

1. Create a new Python file in the `fenics/models/` directory:

```python
# fenics/models/my_model.py
import torch.nn as nn
import torch.nn.functional as F
from fenics.models import ModelBase, ModelFactory

class MyModel(ModelBase):
    def __init__(self):
        super().__init__()
        # Define your model architecture
        
    def forward(self, x):
        # Implement forward pass
        return x

# Register the model
ModelFactory.register_model('my_model', MyModel)
```

2. Import your file and use it:

```python
import fenics.models.my_model

# Use in configuration
# model_type: my_model
```

### Adding Custom Client Selection Strategies

1. Create a new Python file in the `fenics/client_selection/strategies/` directory:

```python
# fenics/client_selection/strategies/my_strategy.py
import random
from typing import List, Optional
import logging

def select_clients_my_strategy(nodes: List[int], 
                              num_participants: int, 
                              logger: Optional[logging.Logger] = None) -> List[int]:
    """Custom selection strategy."""
    logger = logger or logging.getLogger()
    # Implement your selection logic
    return selected_nodes
```

2. Register and use it:

```python
from fenics.client_selection.factory import SelectionFactory
from fenics.client_selection.strategies.my_strategy import select_clients_my_strategy

# Register the strategy
SelectionFactory.register_strategy('my_strategy', select_clients_my_strategy)
```

### Adding Custom Attack Types

1. Create a new Python file in the `fenics/attack/attack_types/` directory:

```python
# fenics/attack/attack_types/my_attack.py
from fenics.attack.attack_types.base import Attack
from fenics.attack.attack_factory import AttackFactory

class MyAttack(Attack):
    def __init__(self, node_id, logger=None):
        super().__init__(node_id, logger)
        
    def execute(self, model):
        # Implement your attack logic
        return model.state_dict()

# Register the attack
AttackFactory.register_attack('my_attack', MyAttack)
```

2. Import and use your attack:

```python
import fenics.attack.attack_types.my_attack

# Use in configuration
# attacks: [my_attack]
```

### Adding Custom Aggregation Strategies

1. Create a new Python file in the `fenics/aggregation/` directory:

```python
# fenics/aggregation/my_strategy.py
import torch
from fenics.aggregation.base import AggregationStrategy
from fenics.aggregation.factory import AggregationFactory

class MyAggregationStrategy(AggregationStrategy):
    def aggregate(self, models_state_dicts, data_sizes):
        # Implement your aggregation logic
        return aggregated_state_dict

# Register the strategy
AggregationFactory.register_strategy('my_aggregation', MyAggregationStrategy)
```

2. Import and use your strategy:

```python
import fenics.aggregation.my_strategy

# Use in your code
agg_strategy = AggregationFactory.get_strategy('my_aggregation')
```

### Adding Custom Communication Protocols

1. Create a new Python file in the `fenics/communication/` directory:

```python
# fenics/communication/my_protocol.py
from fenics.communication.base import CommunicationProtocol
from fenics.communication.factory import ProtocolFactory

class MyProtocol(CommunicationProtocol):
    def exchange(self, nodes, G, local_models, executor):
        # Implement your exchange logic
        pass

# Register the protocol
ProtocolFactory.register_protocol('my_protocol', MyProtocol)
```

2. Import and use your protocol:

```python
import fenics.communication.my_protocol

# Use in configuration
# protocol: my_protocol
```

---

## Examples

Here's a step-by-step example to demonstrate how to set up and run a simulation using a custom network topology.
    
### 1. Define a Custom Topology
    
Create a file named `topology.edgelist` with the following content:
    
```
0 1
1 2
1 4
2 3
3 5
3 6
4 5
5 6
5 7
7 8
8 9
```
    
### 2. Configure `config.yaml`
    
Create or update your `config.yaml` to include the custom topology.
    
```yaml
simulations:
  simulation1:
    rounds: 5
    epochs: 2
    num_nodes: 10
    num_attackers: 2
    attacker_nodes: [3, 7]
    attacks: ['poison', 'delay']
    use_attackers: true
    participation_rate: 0.6
    topology: custom
    topology_file: topology.edgelist
    max_attacks: 3
    gossip_steps: 4
    protocol: gossip
    alpha: 0.5
    model_type: cnn
```
    
### 3. Run the Simulation
    
1. **Launch the Fenics Shell:**
    
    ```bash
    fenics
    ```
    
2. **Setup the Simulation Using `config.yaml`:**
    
    ```bash
    setup --config config.yaml --simulation_name simulation1
    ```
    
3. **Execute the Simulation:**
    
    ```bash
    run
    ```
    
4. **Monitor Outputs:**
    
    - **Logs:** Check the `results/simulation_log.txt` file for detailed logs.
    - **Plots:** Generated plots will be saved as PDF files in the `results` directory.
    - **Simulation Results:** Ensure that attacker nodes (3 and 7) perform their designated attacks, and observe how the network handles these adversarial behaviors.
    
---

## Contributing

Contributions are welcome! If you'd like to enhance Fenics, please follow these steps:
    
1. **Fork the Repository:**
    
    Click the "Fork" button on the repository's GitHub page to create your own copy.
    
2. **Clone Your Fork:**
    
    ```bash
    git clone https://github.com/yourusername/fenics.git
    cd fenics
    ```
    
3. **Create a New Branch:**
    
    ```bash
    git checkout -b feature/your-feature-name
    ```
    
4. **Make Your Changes:**
    
    Implement your desired features or fixes.
    
5. **Commit Your Changes:**
    
    ```bash
    git add .
    git commit -m "Add feature XYZ"
    ```
    
6. **Push to Your Fork:**
    
    ```bash
    git push origin feature/your-feature-name
    ```
    
7. **Create a Pull Request:**
    
    Navigate to the original repository and create a pull request from your forked repository.
    
---

## License

This project is licensed under the [MIT License](LICENSE).

---
## Acknowledgments

- This project was part of a course project (DAT-300: Data-diven support for cyberphysical systems) at Chalmers University of Technology under the supervision of Romaric Duvignau and Carla Fabiana Chiasserini.

---

## Contact

For any questions or feedback, please contact [Shubham Saha](mailto:shuvsaha7@gmail.com) or [Sifat Nawrin Nova](mailto:nawrinnova04@gmail.com).
