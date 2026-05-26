import csv
from datasets import load_dataset
from tqdm.auto import tqdm
# Download the LAMA dataset
dataset = load_dataset("janck/bigscience-lama", "Trex")
data = dataset["test"]

# Define the path to save the converted dataset
output_file = "/home/mmahaut/projects/parrots/data/lama.csv"

# Open the output file in write mode
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow([
        "dataset", "rel_type", "rel_num", "augmentation", "subj_position", "subject", 
        "template", "to_send", "num_tokens_sub", "unrelated_origin", "corr_answer", 
        "repetition", "num_tokens_obj"
    ])

    # Write each example to the file
    for example in tqdm(data):
        # columns are dataset,rel_type,rel_num,augmentation,subj_position,subject,template,to_send,num_tokens_sub,unrelated_origin,corr_answer,repetition,num_tokens_obj
        dataset = "LAMA_Trex"
        rel_type = "Hu_original"
        rel_num = example["predicate_id"]
        augmentation = "original"
        subj_position = ""
        subject = example["sub_label"]
        template = example["template"]
        
        # check [Y] ends the template
        if template.endswith("[Y]") or template.endswith("[Y] ."):
            to_send = template.replace("[X]", example["sub_label"]).split("[Y]")[0]
        else:
            # skip if [Y] is not at the end
            continue
        num_tokens_sub = ""
        unrelated_origin = ""
        corr_answer = example["obj_label"]
        repetition = ""
        num_tokens_obj = ""
        writer.writerow([
            dataset, rel_type, rel_num, augmentation, subj_position, subject, template,
            to_send, num_tokens_sub, unrelated_origin, corr_answer, repetition, num_tokens_obj
        ])

print("Dataset conversion completed!")