import pytest
from pathlib import Path
import pandas as pd
import json
import seaborn as sns
import numpy as np
from ..rraa_graph_heatmap import process_file, collect_files, group_columns

import matplotlib.pyplot as plt

@pytest.fixture
def mock_json_file(tmp_path):
    data = {
        "params": {
            "model_name": "test_model",
            "revision": "1",
            "n_cycles": 1,
            "cycle_size": 1,
            "dataset_size": 100,
            "batch_size": 10,
            "idx": 0
        },
        "layer.0.head.0": 10,
        "layer.0.head.1": 20,
        "layer.1.head.0": 30,
        "layer.1.head.1": 40
    }
    file_path = tmp_path / "test.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

def test_process_file(mock_json_file):
    result = process_file(mock_json_file)
    assert result is not None
    assert "params" in result

def test_collect_files(mock_json_file, tmp_path):
    base_folder = tmp_path
    result = collect_files(base_folder)
    assert len(result) == 1
    assert "params" in result[0]

def test_group_columns():
    att_dict = {
        "params": {
            "model_name": "test_model",
            "revision": "1",
            "n_cycles": 1,
            "cycle_size": 1,
            "dataset_size": 100,
            "batch_size": 10,
            "idx": 0
        },
        "layer.0.head.0": 10,
        "layer.0.head.1": 20,
        "layer.1.head.0": 30,
        "layer.1.head.1": 40
    }
    result = group_columns(att_dict)
    assert len(result) == 4
    assert all("layer" in r for r in result)
    assert all("head" in r for r in result)
    assert all("value" in r for r in result)

def test_axes_shape():
    df = pd.DataFrame({
        "model_name": ["test_model"] * 4,
        "layer": [0, 0, 1, 1],
        "head": [0, 1, 0, 1],
        "value": [10, 20, 30, 40],
        "cycle_size": [1, 1, 1, 1],
        "idx": [0, 0, 0, 0],
        "dataset_size": [100, 100, 100, 100],
        "Checkpoint nº": [1, 1, 1, 1]
    })
    sns.set_theme(style="whitegrid", palette="colorblind")
    model_name = "test_model"
    model_df = df[df["model_name"] == model_name]
    num_xplots = len(model_df["idx"].unique())
    num_yplots = len(model_df["cycle_size"].unique())
    fig, axes = plt.subplots(num_yplots, num_xplots, figsize=(10 * num_xplots, 10 * num_yplots), sharey=True)
    
    if num_xplots == 1 and num_yplots == 1:
        axes = np.array([[axes]])
    elif num_xplots == 1:
        axes = np.array([[ax] for ax in axes])
    elif num_yplots == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(num_yplots, num_xplots)
    
    assert axes.shape == (num_yplots, num_xplots)
    plt.close(fig)