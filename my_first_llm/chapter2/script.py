import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
"Hello, do you like tea Guilherme? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

word = "Akwirw ier"
word2 = "Guilherme"

intergers_word = tokenizer.encode(word2)
print(intergers_word)

# o método tokenizer.decode() espera uma sequência de inteiros (lista), e não um único inteiro
for w in intergers_word:
  print(f"{w}: {tokenizer.decode([w])}")

