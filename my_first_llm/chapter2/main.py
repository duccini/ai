from data_loader import download_and_read_text
from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2, create_vocabulary

# baixa o texto do github, cria um arquivo .txt e carrega o texto 
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
raw_text = download_and_read_text(url)

# create the vocabulary
vocab = create_vocabulary(raw_text)
print("Vocabulary size:", len(vocab))

'''
for i, item in enumerate(vocab.items()):
  print(item)
  if i >= 50:
    break
'''


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Encode text to token IDs:", ids)
print("Decode token IDs to text:", tokenizer.decode(ids))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text2 = " <|endoftext|> ".join((text1, text2))
print(text2)

tokenizer2 = SimpleTokenizerV2(vocab)
print(tokenizer2.encode(text2))
print(tokenizer2.decode(tokenizer2.encode(text2)))