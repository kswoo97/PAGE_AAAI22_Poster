import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from pysmiles import read_smiles
import networkx as nx
import torch.nn.functional as F
# Utility functions
from graph_classification_utils import *
# GNN models
from graph_classification_gnns import *

# Explainer
from graph_classification_prototype_explainer import *

# Benchmark explainer
from graph_classification_XGNN import *

"""
Finding house shape data
Yonsei App.Stat. Sunwoo Kim
There are only three features
"""

"""
Generate data
"""

def testing_faithfulness() :
    # Model accuracy and attribution score should be proportional to each other

def testing_consistency() :
    # Evaluation should be always consistent under various models