import pandas as pd
from pathlib import Path
import re
from rouge_score import rouge_scorer

if __name__ == "__main__":
    base_path="/home/mmahaut/projects/parrots/outputs"

    def calculate_rouge_l(gen_sentence, ref_sentence):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(ref_sentence, gen_sentence)
        if scores['rougeL'].fmeasure is None:
            return 0
        return scores['rougeL'].fmeasure  # Return F1 score for ROUGE-L

    for path in sorted(Path(base_path).glob("*/*/slot_filling_results.csv")):
        df = pd.read_csv(path)
        if "entire_subject_in_generated" in df.columns:
            df=df.drop(columns=["entire_subject_in_generated","partial_subject_in_generated"])
        if "entire_subject_in_generated" not in df.columns:
            df["entire_subject_in_generated"] = df.apply(lambda x:bool(re.search(f"\\b{re.escape(str(x['subject']))}\\b", re.escape(str(x["generated"])))), axis=1)
            df["partial_subject_in_generated"] = df.apply(lambda x: any([bool(re.search(f"\\b{str(sub)}\\b", re.escape(str(x["generated"])))) for sub in re.escape(str(x["subject"])).split()]), axis=1)
            df["rougel_generated_vs_prompt"] = df.apply(lambda x: calculate_rouge_l(re.escape(str(x["generated"])),re.escape(str(str(x["template"].replace("[X]","").replace("[Y]",""))))), axis=1)
            if "subject_in_generated" in df.columns:
                # delete
                df.drop(columns=["subject_in_generated"], inplace=True)
            df.to_csv(path, index=False)

        if "zero " in df["augmentation"].unique():
            # rename
            df["augmentation"] = df["augmentation"].apply(lambda x: x.replace("zero ", "zero")) 
            df.to_csv(path, index=False)

        print("\n",path.parent.stem, "parrot" if "human" in str(path.parent) else "lama")
        # find all possible values for column "augmentation"
        # print("augmentation: ", df["augmentation"].unique())
        acc_df=pd.DataFrame()
    
        for augm in df["augmentation"].unique():
            # print("augmentation: ", augm)
            # print("exact_match: ", df[df["augmentation"] == augm]["exact_match"].mean())
            # print("direct_follow: ", df[df["augmentation"] == augm]["direct_follow"].mean())
            # print("nli_factual: ", df[df["augmentation"] == augm]["nli_factual"].mean())
            # check for apparition of "subject" in "generated"
            augm = {
                "augmentation": augm,
                "exact_match": df[df["augmentation"] == augm]["exact_match"].mean(),
                "direct_follow": df[df["augmentation"] == augm]["direct_follow"].mean(),
                "nli_factual": df[df["augmentation"] == augm]["nli_factual"].mean(),
                "entire_subject_in_generated": df[df["augmentation"] == augm]["entire_subject_in_generated"].mean(),
                "partial_subject_in_generated": df[df["augmentation"] == augm]["partial_subject_in_generated"].mean(),
                "rougel_generated_vs_prompt": df[df["augmentation"] == augm]["rougel_generated_vs_prompt"].mean(),
            }
            acc_df = pd.concat([acc_df, pd.DataFrame(augm, index=[0])])
        print(acc_df)

        