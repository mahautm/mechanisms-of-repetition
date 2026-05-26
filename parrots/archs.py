from transformers import AutoModelForCausalLM, AutoTokenizer

def get_model(model_name, no_grad=True, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if no_grad:
        for p in model.parameters():
            p.requires_grad = False
    return model, tokenizer


def get_tokenizer(model_name, **kwargs):
    t = AutoTokenizer.from_pretrained(model_name, padding_side="left", **kwargs)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t
