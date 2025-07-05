# Dataset Description

This dataset consists of graphs and their corresponding textual descriptions. The goal is to generate realistic graphs that reflect the structure described in natural language.


## Directory Structure

The dataset is organized as follows:

data/
├── train/
│ ├── graphs/ # Graph files (.edgelist and .graphml)
│ └── descriptions/ # Corresponding textual descriptions (.txt)
├── valid/
│ ├── graphs/
│ └── descriptions/
└── test/
└── test.txt # Only descriptions (no ground-truth graphs)


## Details

1. **Training Set**
   - Contains 8,000 samples
   - Each sample has:
     - A graph file in `data/train/graphs/`, available in both `.edgelist` and `.graphml` formats.
     - A corresponding description file in `data/train/descriptions/`, in `.txt` format.
   - Example:
     - Graph: `data/train/graphs/graph_1.edgelist`
     - Description: `data/train/descriptions/graph_1.txt`

2. **Validation Set**
   - Contains 1,000 samples
   - Same structure as the training set.

3. **Test Set**
   - Contains 1,000 textual descriptions in a single file: `data/test/test.txt`
   - Each line in `test.txt` is a separate graph description.
   - Important: **Do not shuffle this file.** The order is used for evaluation.

---

## Download Link

Dataset can be downloaded from:  
https://drive.google.com/file/d/1Ey54FhVnIUlryhV_AwUFykp4mdjUvcul/view?usp=sharing

After downloading and unzipping, place the `train/`, `valid/`, and `test/` folders inside your `data/` directory.


## Notes

- The descriptions specify structural properties such as:
  - Number of nodes
  - Number of connected components
  - Clustering coefficient
  - Degree distribution
  - Diameter, etc.

- Your model must generate graphs that match these properties as closely as possible.

