with open("generate_alluvial_data.py", "r") as f:
    text = f.read()

text = text.replace("""        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            revision=args.revision,
            
            
        )""", """        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            revision=args.revision,
            torch_dtype=torch.float16
        )""")
text = text.replace("""        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            
            
        )""", """        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16
        )""")

if "tokenizer.padding_side = 'left'" not in text:
    text = text.replace("if tokenizer.pad_token is None:", "tokenizer.padding_side = 'left'\n    if tokenizer.pad_token is None:")

with open("generate_alluvial_data.py", "w") as f:
    f.write(text)
