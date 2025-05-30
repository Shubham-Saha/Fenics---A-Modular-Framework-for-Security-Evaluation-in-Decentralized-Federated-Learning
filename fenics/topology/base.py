# fenics/topology/base.py

import networkx as nx
from abc import ABC, abstractmethod
from typing import List


def create_nodes(num_nodes: int) -> List[int]:
    """
    Create a list of node indices.
    
    Args:
        num_nodes: Number of nodes
        
    Returns:
        List of node indices from 0 to num_nodes-1
    """
    return list(range(num_nodes))  # Nodes are represented by integer indices


class TopologyBase(ABC):
    """
    Base class for all network topologies.
    """
    
    def __init__(self, num_nodes: int):
        """
        Initialize the topology.
        
        Args:
            num_nodes: Number of nodes in the network
        """
        self.num_nodes = num_nodes
    
    @abstractmethod
    def build(self) -> nx.Graph:
        """
        Build the network topology.
        
        Returns:
            NetworkX graph representing the topology
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of the topology.
        
        Returns:
            Name of the topology
        """
        return self.__class__.__name__.replace('Topology', '')