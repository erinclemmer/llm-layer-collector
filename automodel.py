import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

login()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

num_tokens = 10

input_ids = tokenizer("The quick brown fox jumps over the ")['input_ids']
for _ in range(num_tokens):
    output = model(torch.tensor([input_ids]), use_cache=False)

    probs = torch.softmax(output.logits[:, -1, :][0], dim=-1)
    top = int(torch.topk(probs, 1).indices[0])

    input_ids.append(top)
    print(tokenizer.decode(input_ids))