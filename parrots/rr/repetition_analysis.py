import re
import pandas as pd
from pathlib import Path

def repetitions(s):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(s):
        if len(match.group(1)) > 1 and len(match.group(0)) / len(match.group(1)) > 1:
            yield (match.group(1), len(match.group(0)) / len(match.group(1)))
       
def repetitions_with_space(s):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(s):
        if ' ' in match.group(1):
            yield (match.group(1), len(match.group(0)) / len(match.group(1)))

def repeating_words(s):
    words = re.findall(r'\b\w+\b', s)
    r = re.compile(r"(\b\w+\b)(?:\s+\1)+")
    for match in r.finditer(s):
        word = match.group(1)
        count = len(re.findall(r'\b\w+\b', match.group(0))) / len(re.findall(r'\b\w+\b', word))
        if count > 1:
            yield (word, count)

def main(data_file):
    df = pd.read_csv(data_file)
    df["repetitions"] = df["generated"].apply(lambda x: list(repetitions(x)))
    df["repetitions_with_space"] = df["generated"].apply(lambda x: list(repetitions_with_space(x)))
    df["repeating_words"] = df["generated"].apply(lambda x: list(repeating_words(x)))
    return df

if __name__ == "__main__":
    file_path = "/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results.csv"
    df=pd.read_csv(file_path)
    df["repetitions"] = df["generated"].apply(lambda x: [x[1] for x in list(repeating_words(str(x)))])
    df["repetitions"]=df["repetitions"].apply(lambda x: max(x) if len(x)>0 else 0)

    df.to_csv(file_path, index=False)
    # df=df[df["augmentation"]=="unrelated"]
    print(df[df["repetitions"]>0]["repetitions"].mean(), len(df[df["repetitions"]>0]), len(df), len(df[df["repetitions"]>0])/len(df))


    # check if subject is repeated in generated text
    df["s_repetitions"] = df["generated"].apply(lambda x: [x[0] for x in list(repeating_words(str(x)))]) # more than one repetition of the subject
    df["s_repetitions"]=df.apply(lambda x: x["subject"] in x["s_repetitions"] if x["s_repetitions"] else False, axis=1)
    print(df["s_repetitions"].mean(), len(df[df["s_repetitions"]>0]), len(df), len(df[df["s_repetitions"]>0])/len(df))

