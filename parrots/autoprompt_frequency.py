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



if __name__ == "__main__":
    file_path = "/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_autoprompts_opt1_3b_lama_parrot_list_v1_sf/slot_filling_results.csv"
    df=pd.read_csv(file_path)
    # count most frequent generations
    df["count_gen"] = df["generated"].map(df["generated"].value_counts())
    df["biggest_duplicate"] = df.groupby("template")["count_gen"].transform("max") #/ df.groupby("template")["generated"].transform("count")
    # extract list of subjects with maximum count_gem
    subs={}
    cov={}
    for temp in df["template"].unique():
        max_gen_df=df[df["count_gen"]==df["biggest_duplicate"]]
        subs[temp]=max_gen_df[max_gen_df["template"]==temp]["subject"].tolist()
        for temp_b in subs.keys():
            if temp == temp_b:
                cov[f"{temp} & {temp_b}"] = 1
            else:
                cov[f"{temp} & {temp_b}"] = len([t for t in subs[temp_b] if t in subs[temp]])
    for k in cov.keys():
        print(k, cov[k])
        

    
    # print a co matrix where the number of matching subjects for each template pair appears
    
    
    # for all unique "template", see how many different generations there are, normalize per template
    df["original_gens"] = df.groupby("template")["generated"].transform("nunique") #/ df.groupby("template")["generated"].transform("count")
    # count duplicate rows
    df["n_sentences"] = df.groupby("template")["generated"].transform("count")
    df["n_gens"] = df.groupby("template")["generated"].transform("count")
    print(df.drop_duplicates(["original_gens"])[["template","original_gens","biggest_duplicate", "n_gens"]].sort_values(by="biggest_duplicate",ascending=False))

    # print(df[["template","count_gen_per_template"]].drop_duplicates().sort_values(by="count_gen_per_template", ascending=True))
    # some templates have a lot less generations than others
    # group by template, and count the number of unique generations
    # print(df.groupby("template")["generated"].nunique().sort_values(ascending=True))





