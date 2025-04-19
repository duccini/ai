import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Tokenização com BPE (Byte Pair Encoding)
tokenizer = tiktoken.get_encoding("gpt2")

# PyTorch Class
class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(txt) # 1

    # 2. Criação de pares (entrada-alvo) para treino
    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i + max_length]
      target_chunk = token_ids[i + 1: i + max_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
  

# Data Loader that use GPTDatasetV1
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
  )
  return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

# 3. Embedding dos tokens
# Cada token é mapeado para um vetor contínuo de dimensão 256.
# Se inputs.shape = (8, 4), ou seja, 8 sequências (pq 8 sequencias?) com 4 tokens cada:
# token_embeddings.shape = (8, 4, 256)
# Isso é chamado de token embeddings — eles capturam propriedades semânticas dos tokens.
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

'''
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)


first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch) 

inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
'''

dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length,
stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
#print("Token IDs:\n", inputs)
#print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
#print(token_embeddings.shape)

# 4. Positional Embeddings
# Transformers não têm recorrência nem convoluções, então precisamos indicar a posição dos tokens explicitamente.
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
#print(pos_embeddings.shape)

# 5. Soma: token + posição
# Agora temos vetores com conteúdo + posição, prontos para entrar no modelo.
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


'''
1. Por que precisamos de "token embeddings" e "positional embeddings"?
- Token embeddings codificam o que está sendo dito (semântica).
- Positional embeddings codificam a ordem das palavras (posição na sequência).

Transformers processam tudo em paralelo, então sem posição, a ordem se perde — "Eu amo você" vira indistinguível de
 "Você amo eu".

2. Podemos calcular quantos parâmetros esse modelo tem?
Sim! No estado atual (apenas embeddings), o número de parâmetros é:

📦 Token embedding:
vocab_size = 50257
output_dim = 256
token_embedding_params = 50257 x 256 = 12,653,312

📦 Positional embedding:
context_length = 4
pos_embedding_params = 4 * 256 = 1,024

✅ Total atual: 12,653,312 + 1,024 = 12,654,336 parâmetros


'''