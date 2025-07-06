import os
from tqdm import tqdm
import random
import re
import requests
import zipfile
import os
import shutil
random.seed(32)


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file, "r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats


def load_data(repo_url, download_dir):
    try:
        zip_url = f"{repo_url}/archive/refs/heads/main.zip"  # Update branch if not "main"
        response = requests.get(zip_url)
        response.raise_for_status()

        # Save zip file
        zip_path = os.path.join(download_dir, "repo.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

        # Clean up zip file
        os.remove(zip_path)

        # Locate the folder that was created after extracting the repo
        repo_folder_name = os.path.basename(repo_url) + "-main"
        repo_folder_path = os.path.join(download_dir, repo_folder_name)

        # Path to the new DATA folder
        data_folder_path = os.path.join(download_dir, "data")
        os.makedirs(data_folder_path, exist_ok=True)

        # Path to the data.zip inside the extracted repo folder
        data_zip_path = os.path.join(repo_folder_path, "data.zip")

        # Extract data.zip into the DATA folder
        with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
            zip_ref.extractall(data_folder_path)

        # Remove the repo folder after extracting data.zip
        shutil.rmtree(repo_folder_path)

        print(f"Repository and data extracted successfully. data folder created at {data_folder_path}")
    except Exception as e:
        print(f"Error: {e}")
