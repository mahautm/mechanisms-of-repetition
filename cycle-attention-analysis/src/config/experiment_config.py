"""Configuration for attention distribution experiments."""

import os

# Model configuration
MODEL_CONFIG = {
    'model_name': 'EleutherAI/pythia-1.4b',
    'revision': None,
    'use_bfloat16': False,
}

# Data configuration  
DATA_CONFIG = {
    'seed': 42,
    'n_samples': 100,
    'max_length': 256,
    'max_new_tokens': 100,
    'batch_size': 1,
}

# Analysis configuration
ANALYSIS_CONFIG = {
    'layer': 12,
    'n_cycles': 3,
    'output_dir': 'data/results',
}

# Visualization configuration
VIZ_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
}

class ExperimentConfig:
    def __init__(self):
        # Model parameters
        self.model_name = "EleutherAI/pythia-1.4b"
        self.revision = None
        self.use_bfloat16 = False
        
        # Data parameters
        self.data_path = os.path.join("data", "results")
        self.n_samples = 5000
        self.batch_size = 1
        self.max_length = 256
        self.max_new_tokens = 100
        
        # Analysis options
        self.n_cycles = 0
        self.no_head_analysis = False
        
        # Output paths
        self.output_dir = "data/results"
        self.plot_dir = "plots"
        
    def display_config(self):
        print("Experiment Configuration:")
        print(f"Model Name: {self.model_name}")
        print(f"Data Path: {self.data_path}")
        print(f"Number of Samples: {self.n_samples}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Max Length: {self.max_length}")
        print(f"Max New Tokens: {self.max_new_tokens}")
        print(f"Number of Cycles: {self.n_cycles}")
        print(f"No Head Analysis: {self.no_head_analysis}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Plot Directory: {self.plot_dir}")