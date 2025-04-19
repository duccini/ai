import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()
  enc_text = tokenizer.encode(raw_text)
print("Total number of tokens with BPE tokenizer:", len(enc_text))

enc_sample = enc_text[50:]

# The context size determines how many tokens are included in the input.
context_size = 4

# input tokens
x = enc_sample[:context_size]

# targets. Why targets if LLM predict one word at time?
# if we know the targets in advanced what is the use of this?
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
  context = enc_sample[:i]
  desired = enc_sample[i] # how access 2nd item?
  print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))