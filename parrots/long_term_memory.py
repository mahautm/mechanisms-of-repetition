from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer
    model_name = "mistralai/Mistral-7B-v0.3" # "EleutherAI/pythia-1.4B"
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cycle = "dogs are red"
    context = "Matéo was born in Ambilly"
    query = "Where was Matéo born?"
    answer = "Ambilly"
    # Load the Pile dataset
    dataset = load_dataset("JeanKaddour/minipile")
    subset = dataset["train"].shuffle(seed=0)
    subset = subset.select(range(len(subset)//10))

    # params
    max_cycles = 1000
    cycle_score = {}
    pile_score = {}
    scrambled_score = {}

    # precompute cycle_len
    cycle_len = len(tokenizer(cycle)["input_ids"])
    for cycle_idx in tqdm(range(1, max_cycles, 10), desc="Processing Cycles"):
        # Generate the next token
        input = context + cycle * cycle_idx + query
        input = tokenizer(input, return_tensors="pt").to(device)
        tqdm.write(f"Input1: {input['input_ids'].shape}")
        output = model.generate(**input, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        # Check if the answer is in the output
        cycle_score[cycle_idx] = answer in output

        # Compute the score for the Pile dataset
        random_sentence = subset[cycle_idx % len(subset)]["text"]
        # only keep the same amount of tokens as cycle * cycle_idx
        cycle_tok_len = cycle_len * cycle_idx
        tok_random_sentence = tokenizer(random_sentence, return_tensors="pt")["input_ids"]
        if tok_random_sentence.shape[1] > cycle_tok_len:
            tok_random_sentence = tok_random_sentence[:, :cycle_tok_len]
        else:
            tok_random_sentence = torch.cat((tok_random_sentence, tokenizer(subset[0]["text"], return_tensors="pt")["input_ids"][:cycle_tok_len - len(tok_random_sentence)]), dim=1)
        random_sentence = tokenizer.decode(tok_random_sentence.squeeze(), skip_special_tokens=True)
        input = context + random_sentence + query
        input = tokenizer(input, return_tensors="pt").to(device)
        tqdm.write(f"Input2: {input['input_ids'].shape}")
        output = model.generate(**input, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        pile_score[cycle_idx] = answer in output

        # Compute the score for the scrambled dataset
        scrambled_sentence_tokens = torch.randint(0, tokenizer.vocab_size, (cycle_tok_len,))
        scrambled_sentence = tokenizer.decode(scrambled_sentence_tokens, skip_special_tokens=True)
        input = context + scrambled_sentence + query
        input = tokenizer(input, return_tensors="pt").to(device)
        tqdm.write(f"Input3: {input['input_ids'].shape}")
        output = model.generate(**input, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        scrambled_score[cycle_idx] = answer in output

        tqdm.write(f"Cycle {cycle_idx}: {cycle_score[cycle_idx]}, {pile_score[cycle_idx]}, {scrambled_score[cycle_idx]}")

    print(cycle_score)
    print(pile_score)
    print(scrambled_score)

    # Plot the scores
    plt.figure(figsize=(12, 6))
    plt.plot(list(cycle_score.keys()), list(cycle_score.values()), label="Cycle Score")
    plt.plot(list(pile_score.keys()), list(pile_score.values()), label="Pile Score")
    plt.plot(list(scrambled_score.keys()), list(scrambled_score.values()), label="Scrambled Score")
    plt.xlabel("Cycle Index")
    plt.ylabel("Score")
    plt.title("Scores over Cycles")
    plt.legend()
    plt.savefig("scores.png")

if __name__ == "__main__":
    main()