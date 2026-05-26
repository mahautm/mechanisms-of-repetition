import pandas as pd
import random
from transformers import pipeline
import torch
from scipy.stats import chi2_contingency
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if a GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load a pre-trained language model
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# Define a function to generate text
def llm(prompt, max_length=50, do_sample=True, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length, do_sample=do_sample, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [{"generated_text": generated_text}]

# generate nouns
nouns_set = set()

while len(nouns_set) < 1000:
    rd_nouns = random.sample(nouns_set, 10) if len(nouns_set) > 10 else list(nouns_set)
    prompt = "Liste de noms communs d'un seul mot suivit de féminin ou masculin --> \nNom: table, Genre: féminin\nNom: chapeau, Genre: masculin\nNom: carotte, Genre: féminin\n" + "\n".join([f"Nom: {noun}, Genre: {gender}" for noun, gender in rd_nouns])
    print(prompt)
    text = llm(prompt, max_length=3000, do_sample=True, temperature=0.8)[0]["generated_text"]
    # Split the text into lines
    lines = text.split("\n")
    # Extract the nouns and gender
    for line in lines:
        noun_gender = tuple(line.split(","))
        if len(noun_gender) == 2:
            try:
                noun = noun_gender[0].split(":")[1].strip()
                gender = noun_gender[1].split(":")[1].strip().lower()
                if gender in ["féminin", "masculin"]:
                    nouns_set.add((noun, gender))
                else:
                    print("Invalid gender:", gender)
            except IndexError as e:
                print("Error parsing line:", line, "Error:", e)
        else:
            print("ERR:",line)
    print(f"Generated {len(nouns_set)} nouns, {len(nouns_set) / 1000 * 100:.2f}%, {nouns_set}")

# Convert the set to a list
random_nouns = list(nouns_set)

# Create a DataFrame
df = pd.DataFrame(random_nouns, columns=["Noun", "Gender"])
print(df.head())
# Define a function to classify the noun as more noble or more popular
def classify_noun(noun):
    prompt = f"Is the word '{noun}' semantically more noble or more popular?"
    response = llm(prompt, max_length=1000, do_sample=True, temperature=0.7)[0]["generated_text"]
    # only keep the continuation of the prompt, check if it is more noble or popular
    response = response.split(prompt)[1]
    return "noble" if "noble" in response else "popular"
# Apply the function to each noun in the DataFrame
df["Classification"] = df["Noun"].apply(classify_noun)
print(df["Classification"].value_counts())

# Count the occurrences of each gender
gender_counts = df["Gender"].value_counts()

# Calculate the percentage of each gender
gender_percentages = (gender_counts / len(df)) * 100

# Print the percentages
print("Percentage of Feminine Nouns: {:.2f}%".format(gender_percentages.get("féminin", 0)))
print("Percentage of Masculine Nouns: {:.2f}%".format(gender_percentages.get("masculin", 0)))

# percentage of popular and noble nouns per gender and statistical significance
popular_nouns = df[df["Classification"] == "popular"]
noble_nouns = df[df["Classification"] == "noble"]

print(f"percentage of popular nouns: {len(popular_nouns) / len(df) * 100:.2f}%")
print(f"percentage of noble nouns: {len(noble_nouns) / len(df) * 100:.2f}%")

print(f"percentage of feminine popular nouns: {len(popular_nouns[popular_nouns['Gender'] == 'féminin']) / len(df) * 100:.2f}%")
print(f"percentage of masculine popular nouns: {len(popular_nouns[popular_nouns['Gender'] == 'masculin']) / len(df) * 100:.2f}%")

print(f"percentage of feminine noble nouns: {len(noble_nouns[noble_nouns['Gender'] == 'féminin']) / len(df) * 100:.2f}%")
print(f"percentage of feminine noble nouns: {len(noble_nouns[noble_nouns['Gender'] == 'masculin']) / len(df) * 100:.2f}%")

# Create a contingency table
contingency_table = pd.crosstab(df["Gender"], df["Classification"])

# Perform the chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the results
print(f"Chi-squared test statistic: {chi2}")
print(f"p-value: {p}")

# Determine if the result is significant
alpha = 0.05
if p < alpha:
    print("The difference in classifications between genders is statistically significant.")
else:
    print("The difference in classifications between genders is not statistically significant.")

print(df.head())