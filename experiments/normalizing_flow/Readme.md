# Graph Normalizing Flows (GnF)

This repository implements **Graph Normalizing Flows (GnF)** — a graph generative model composed of an autoencoder followed by a normalizing flow. The goal is to encode input graphs into a latent space and decode or generate new graphs by sampling from this space.

---

## 📁 Project Structure

### `autoencoder.py`

Defines the main model architecture for Graph Normalizing Flows.

#### Components:

- **AutoEncoder**: Combines an encoder and a decoder.
  - `encode(data)`: Encodes input graphs into latent vectors.
  - `decode(x_g)`: Decodes latent vectors into adjacency matrices.
  - `loss_function(data)`: Computes L1 loss between predicted and ground truth adjacency matrices.

- **GIN (Graph Isomorphism Network)**:
  - Multi-layer GIN-based encoder with dropout and batch normalization.
  - Aggregates node features and outputs latent graph representation.

- **GATModel (Graph Attention Network)**:
  - Alternative to GIN using attention-based layers.
  - *Note*: A typo in the `GATModel` class should be corrected (`self.conv.append` → `self.convs.append`).

- **Decoder**:
  - Multi-layer perceptron that outputs upper-triangular edge logits.
  - Applies Gumbel-softmax to generate binary adjacency matrices.

---

### `extract_feat.py`

Utility functions to extract numerical features from raw text.

#### Functions:

- `extract_numbers(text)`: Extracts all integers and floats from a given string using regex.
- `extract_feats(file)`: Opens a file and returns a list of all numerical values extracted from its content.

---

## 🔧 Dependencies

- `torch`
- `torch_geometric`
- `torch.nn`
- `torch.nn.functional`
- `tqdm`
- `re`
- `random`

---

## 📝 Notes

- The decoder generates undirected adjacency matrices using Gumbel-softmax sampling.
- Input graphs must have node features (`data.x`) and edge indices (`data.edge_index`).
- The loss is computed on adjacency matrices (`data.A` should be defined in your dataset).

---
