
# Step 1: Sample sentences
sentences = [
    "Machine learning is interesting",
    "Machine learning uses data"
]

# Step 2: Create an empty vocabulary
vocab = {}

# Step 3: Tokenize and build vocabulary
for sentence in sentences:
    tokens = sentence.lower().split()

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

print("Vocabulary:")
print(vocab)

# Step 4: Convert sentences into token IDs
for sentence in sentences:
    tokens = sentence.lower().split()

    token_ids = [vocab[token] for token in tokens]

    print("\nSentence:", sentence)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
