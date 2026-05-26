"""
data preparation script for generating and saving datasets for self-confidence tasks.
"""
import typer

import datasets
import pandas as pd
from datasets import load_dataset, load_from_disk
import numpy as np
import random
from collections import defaultdict
import uuid


def format_template_trex(example):
    """
    Basic Lama format
    """
    fact = (
        example["template"]
        .replace("[X]", example["sub_label"])
        .replace("[Y]", example["obj_surface"])
    )
    return {"text": fact, "is_factual": True}

class FalseTrex:
    def __init__(self, dataset, paraphrase_type=None):
        """
        Main class to format the dataset using both factual from lama trex and false examples.

        Parameters
        ----------
        dataset : datasets.Dataset
            the dataset to use
        paraphrase_type : str
            the type of paraphrase to use. Can be "manual_paraphrase", "mine" or "paraphrase"
        """
        self.dataset = dataset
        self.paraphrase_type = paraphrase_type
        # as dict for faster lookup and replace in formvat_false_trex
        self.dict_dataset = defaultdict(list)
        for idx, example in enumerate(self.dataset):
            if example["template"] in self.dict_dataset.keys():
                self.dict_dataset[example["template"]].append(idx)
            else:
                self.dict_dataset[example["template"]] = [idx]

    def format_false_trex(self, example):
        """
        select a random obj_surface that is not the same as the obj_surface in the example but within the same category
        """
        random_obj_surface = example["obj_surface"]
        while True:
            random_obj_surface = self.dataset[
                np.random.choice(self.dict_dataset[example["template"]], 1)
            ]["obj_surface"][0]
            if random_obj_surface != example["obj_surface"]:
                break
        out_text = (
            example["template"]
            .replace("[Y]", random_obj_surface)
            .replace("[X]", example["sub_label"])
        )
        # generate a new uuid for the new example
        uuid_ = str(uuid.uuid5(uuid.NAMESPACE_DNS, out_text))
        return {"text": out_text, "uuid2": uuid_, "is_factual": False}

    def format_balanced_trex(self, example):
        """
        select a random obj_surface that is not the same as the obj_surface in the example but within the same category
        """
        is_factual = random.choice([True, False])
        if self.paraphrase_type is not None:
            template = get_lpaqa_paraphrase(example, paraphrase_type=self.paraphrase_type)
        else:
            template = example["template"]
        if is_factual:
            out_dict = format_template_trex(
                {
                    "template": template,
                    "sub_label": example["sub_label"],
                    "obj_surface": example["obj_surface"],
                }
            )
            return out_dict

        out_dict = self.format_false_trex(example)
        out_dict["is_factual"] = is_factual
        return out_dict


def save_modified_lama(output_path:str,
    input_path:str=None, 
    excludes_path:str=None,
    num_proc:int=10
    ):
    """
    Downloads the LAMA dataset and saves it in the format for the self-confidence task.
    It generates both True facts and False facts by sampling false completions from the same category.
    Outputs a train and test set.

    Patameters
    ----------
    output_path: str
        path to save the dataset.
    input_path: str
        path to load the dataset from. If None, downloads the dataset.
    num_proc: int
        number of processes to use for multiprocessing.
    """
    if input_path is None:
        dataset = load_dataset("lama", "trex", split="train").shuffle(seed=42)
    else:
        dataset = load_from_disk(input_path).shuffle(seed=42)
    # split dataset into train/test
    # filter non_unique ["obj", "subj", "template"]
    dataset = pd.DataFrame(dataset)
    dataset = dataset.drop_duplicates(subset=["obj_label", "template", "sub_label"])
    if excludes_path is not None:
        excludes = pd.read_csv(excludes_path)
        dataset = dataset[~dataset["template"].isin(excludes["template"])]
    dataset = datasets.Dataset.from_pandas(dataset)

    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    false_data_maker = FalseTrex(test_dataset)
    fte_dataset = test_dataset.map(
        false_data_maker.format_false_trex,
        num_proc=num_proc,
        remove_columns=["uuid", "masked_sentence"],
    ).rename_column("uuid2", "uuid")
    tte_dataset = test_dataset.map(
        format_template_trex,
        num_proc=num_proc,
        remove_columns=["masked_sentence"],
    )

    ftr_dataset = train_dataset.map(
        false_data_maker.format_false_trex,
        num_proc=num_proc,
        remove_columns=["uuid", "masked_sentence"],
    ).rename_column("uuid2", "uuid")
    ttr_dataset = train_dataset.map(
        format_template_trex,
        num_proc=num_proc,
        remove_columns=["masked_sentence"],
    )
    # combine both
    te_dataset = datasets.concatenate_datasets([fte_dataset, tte_dataset])
    tr_dataset = datasets.concatenate_datasets([ftr_dataset, ttr_dataset])

    te_dataset.save_to_disk(output_path + "_test")
    tr_dataset.save_to_disk(output_path + "_train")

if __name__ == "__main__":
    typer.run(save_modified_lama)