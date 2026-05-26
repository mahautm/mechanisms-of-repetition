import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    d2 = pd.read_csv("/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/slot_filling_results.csv")
    augm = d2["augmentation"].unique()
    for a in augm:
        _reps=d2[d2["partial_subject_in_generated"]==True]
        _reps=_reps[_reps["unrelated_origin"]!="corpus"]
        r=_reps[_reps["augmentation"]==a]["rel_num"].value_counts()
        # rename as repetition
        r.rename("repetition", inplace=True)
        _no_reps=d2[d2["partial_subject_in_generated"]==False]
        _no_reps=_no_reps[_no_reps["unrelated_origin"]!="corpus"]

        nr=_no_reps[_no_reps["augmentation"]==a]["rel_num"].value_counts()
        df=pd.DataFrame()
        df=pd.concat([r, nr], axis=1).fillna(0)
        # order columns
        df = df[sorted(df.columns)]
        df.plot(kind='bar', stacked=True, color=['steelblue', 'red'])
        plt.title(a)
        plt.savefig(f"/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/{a}_repetition.png")
        plt.clf()

        #boxplot
        bp=d2[d2["augmentation"]==a]
        bp=bp[bp["unrelated_origin"]!="corpus"]
        bp.boxplot("rougel_generated_vs_prompt",by="rel_num",grid=False)
        plt.title(a)
        plt.savefig(f"/home/mmahaut/projects/parrots/outputs/facebook/opt-1.3b_human_lama_parrots_list_v1_sf/{a}_prompt_rouge.png")
        plt.clf()
