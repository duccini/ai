# Build a Large Language Model - Practice

## Chapter 2: Working with text data

📌 Resumo rápido sobre Byte Pair Encoding (BPE)
O algoritmo BPE funciona assim:

Começa com a tokenização caractere a caractere.

Em seguida, faz merges de pares de caracteres mais frequentes com base em um vocabulário treinado.

Isso continua até que a string seja mapeada com os pares mais próximos do vocabulário aprendido.

Palavras desconhecidas não são ignoradas — são divididas em sub-palavras ou até caracteres individuais.

No caso do tiktoken, o modelo GPT-2 usa um vocabulário BPE com cerca de 50 mil tokens.

<br>

---

🧪 Testando "Akwirw ier"

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
word = "Akwirw ier"
integers_word = tokenizer.encode(word)

for w in integers_word:
  print(f"{w}: {tokenizer.decode([w])}")
```

Saída típica (pode variar um pouco dependendo da versão do vocabulário):

```bash
10843: Ak
45960: wir
1101: w
220:  ier
```

Explicação:

10843: Ak – "Ak" é um token conhecido no vocabulário.

45960: wir – "wir" é outro sub-token presente.

1101: w – o último w de "Akwirw" ficou isolado porque a sequência "irw" não existe no vocabulário como um todo.

220: ier – com o espaço no início, " ier" é um token comum.

<br>

---

🔍 Porque "Akwirw ier" é quebrada assim?

1. GPT-2 não conhece a palavra inteira "Akwirw".

2. BPE tenta quebrá-la da melhor forma possível com tokens conhecidos.

3. espaço antes de "ier" é importante: o modelo trata " ier" como um token diferente de "ier".
