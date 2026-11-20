# Text Embedding Model Using PyTorch

A beginner-friendly text embedding and semantic similarity project built with PyTorch.

The goal is to explore how text can be represented as numerical vectors and how a model can learn to rank relevant passages closer to a query than irrelevant ones.

## Overview

This project trains a simple text embedding model using **Triplet Loss**. Given a query, a relevant passage, and an irrelevant passage, the model learns embeddings where:

- The query is more similar to its relevant passage.
- The query is less similar to its irrelevant passage.

Instead of using a pretrained Transformer, the model uses a trainable embedding layer and mean pooling to create sentence representations.

## How It Works

The main data and training pipeline is:

1. **Prepare the dataset:** Load a subset of the MS MARCO dataset from Hugging Face Datasets and extract relevant query-passage pairs.
2. **Split the data:** Create training and test CSV files.
3. **Build a vocabulary:** Assign an integer ID to each word in the training data and add an unknown-token ID.
4. **Create triplets:** Pair each query and relevant passage with a negative passage.
5. **Generate embeddings:** Convert token IDs to vectors using PyTorch's `nn.Embedding`.
6. **Pool word vectors:** Average a sentence's word vectors to produce a fixed-size embedding.
7. **Compare sentences:** Use cosine similarity to compare query and passage embeddings.
8. **Train the model:** Optimize Triplet Loss with the Adam optimizer and save the trained model.

## Model Architecture

```text
Input text
    |
Whitespace tokenization and word-to-ID conversion
    |
Embedding layer (128 dimensions)
    |
Mean pooling
    |
Sentence embedding
    |
Cosine similarity
    |
Triplet loss
    |
Backpropagation and Adam optimizer
```

## Triplet Loss

Each training example contains:

- **Anchor:** A user query
- **Positive:** A relevant passage
- **Negative:** An irrelevant passage

The objective is to make the query-positive similarity exceed the query-negative similarity by a specified margin:

```text
L = max(0, S_neg - S_pos + m)
```

Where `S_pos` is the query-positive similarity, `S_neg` is the query-negative similarity, and `m` is the margin.

## Requirements

Install the project's Python dependencies:

```bash
python -m pip install torch pandas datasets scikit-learn
```

The scripts use the Hugging Face `microsoft/ms_marco` dataset and load the first 5,000 training examples.

## Run the Project

Run these commands from the project directory, in order:

```bash
python prepare_data.py
python split_data.py
python build_vocabulary.py
python prepare_triplets.py
python train_model.py
```

`prepare_data.py` loads the dataset and writes `positive_pairs.csv`. `split_data.py` writes the training and test data. `build_vocabulary.py` writes `vocab.pkl`, and `prepare_triplets.py` writes `triplets_train.csv`. Finally, `train_model.py` trains the model and saves `trained_embedding_model.pth`.

`download_dataset.py` separately loads and explores the same dataset subset; it is optional because `prepare_data.py` loads the dataset itself.

## Training Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Triplet Loss |
| Margin | 0.2 |
| Epochs | 3 |

## Training Results

The model was trained for 3 epochs. Training loss decreased across the reported epochs, indicating that the model was learning to improve the similarity relationship between positive and negative passages.

> Training loss alone does not establish retrieval accuracy. Evaluation on unseen queries is a planned next step.

## Project Structure

```text
embedding_model/
├── build_vocabulary.py
├── convert_to_ids.py
├── download_dataset.py
├── embedding_model.py
├── model.py
├── prepare_data.py
├── prepare_triplets.py
├── split_data.py
├── tokenization.py
├── train_model.py
├── triplet_loss.py
├── positive_pairs.csv
├── train_data.csv
├── test_data.csv
├── triplets_train.csv
├── vocab.pkl
├── trained_embedding_model.pth
├── .gitignore
└── README.md
```

The CSV files, vocabulary, and trained model are generated data/artifacts. The repository's `.gitignore` excludes CSV files; the Python virtual environment and bytecode cache are also excluded.

## Technologies Used

- Python
- PyTorch
- Pandas
- Hugging Face Datasets
- Scikit-learn (for the train/test split)
- Git and GitHub

## Learning Outcomes

Through this project, I explored:

- How text is converted into numerical representations.
- How embedding layers work in PyTorch.
- The difference between context-independent and contextual embeddings.
- How cosine similarity measures relationships between vectors.
- How Triplet Loss helps models learn similarity-based representations.
- How backpropagation and the Adam optimizer update model parameters.
- How training hyperparameters affect model learning.

## Future Improvements

- Evaluate the model on unseen test data.
- Implement retrieval using top-K cosine similarity.
- Experiment with different embedding dimensions and margins.
- Compare performance with pretrained models such as Sentence-BERT.
- Explore more advanced embedding architectures.

## Author

Developed as a hands-on learning project to understand text embeddings, similarity learning, and neural network training using PyTorch.
