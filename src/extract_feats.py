"""
extract_files.py

This module provides utility functions for extracting and processing textual
features from files, including generating dense vector embeddings using a
pre-trained BERT model and reducing their dimensionality with a learned linear layer.

Key functionalities:
- Text embedding generation with BERT
- Dimensionality reduction of embeddings
- Extraction of numerical values from text
- File-based feature extraction

Dependencies:
- transformers (for BERT model and tokenizer)
- torch (for tensor operations and neural networks)
- tqdm (optional, for progress bars)
- re (regular expressions for extracting numbers)
- random (for seeding reproducibility)

Components:

1. EmbeddingReducer(nn.Module):
    - A simple feed-forward neural network layer that reduces the dimensionality
      of embeddings from 768 (BERT base output) to 256.

2. generate_embeddings(text: str) -> torch.Tensor:
    - Tokenizes and encodes input text using a pre-trained BERT model.
    - Averages token embeddings to get a single vector representing the input.
    - Applies dimensionality reduction via the EmbeddingReducer.
    - Returns a 256-dimensional torch tensor embedding.

3. extract_embeddings(file: str) -> torch.Tensor:
    - Reads text content from a specified file.
    - Generates and returns the reduced embedding vector for the file content.

4. extract_numbers(text: str) -> list[float]:
    - Uses regex to find all integer and floating-point numbers in the input text.
    - Returns these numbers as a list of floats.

5. extract_feats(file: str) -> list[float]:
    - Reads the entire text content of the specified file.
    - Extracts and returns all numeric values from the content using `extract_numbers`.

Usage example:

    embedding = extract_embeddings("path/to/description.txt")
    features = extract_feats("path/to/description.txt")

"""
import os
from tqdm import tqdm
import random
import re

random.seed(32)

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


# Load the pre-trained T5 model and tokenizer
# model_name = "t5-base"
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = T5Tokenizer.from_pretrained(model_name)

model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define a dense layer for dimensionality reduction
class EmbeddingReducer(nn.Module):
    """
    Linear layer module to reduce the dimensionality of embeddings from 768 to 256.
    """

    def __init__(self, input_dim=768, output_dim=256):
        super(EmbeddingReducer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, embeddings):
        """
        Forward pass to reduce dimensionality.
        Args:
            embeddings (torch.Tensor): Input embedding tensor of shape (768,) or (batch_size, 768).
        Returns:
            torch.Tensor: Reduced embedding tensor of shape (256,) or (batch_size, 256).
        """
        return self.fc(embeddings)

# Initialize the dimensionality reducer
reducer = EmbeddingReducer(input_dim=768, output_dim=256)

def generate_embeddings(text):
    """
    Generate a fixed-size embedding vector from input text using BERT model.

    Args:
        text (str): Input string to embed.

    Returns:
        torch.Tensor: 256-dimensional embedding vector.
    """
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)


    # Pass the input through the  model to get the embeddings from the encoder
    with torch.no_grad():
        # outputs = model.encoder(input_ids=inputs['input_ids'])
        outputs = model(**inputs)
    
    # The embeddings are stored in the encoder's last hidden state
    last_hidden_states  = outputs.last_hidden_state

    # You can aggregate the embeddings for each token into a single vector (e.g., by averaging)
    embedding_vector = last_hidden_states.mean(dim=1).squeeze()

     # Reduce the dimensionality of embeddings
    reduced_embedding = reducer(embedding_vector)

    return reduced_embedding

def extract_embeddings(file):
    """
    Extract the embedding vector from text read in a file.

    Args:
        file (str): Path to the input text file.

    Returns:
        torch.Tensor: 256-dimensional embedding vector for the file content.
    """

    # Open and read the file
    with open(file, "r") as fread:
        line = fread.read().strip()
    
    # Generate the embedding for the description
    embedding = generate_embeddings(line)
    
    return embedding

def extract_numbers(text):
    """
    Extract all numeric values (integers and floats) from a string.

    Args:
        text (str): Input text containing numbers.

    Returns:
        list of float: List of all numbers found in the text.
    """

    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    """
    Extract numeric features from the content of a file.

    Args:
        file (str): Path to the input file.

    Returns:
        list of float: Numeric values extracted from the file content.
    """
    
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats