import os
import requests
from config.models import MODELS

def download_model(url, save_path):
  if os.path.exists(save_path):
    clobber = input(f"File {save_path} already exists. Do you want to overwrite it? (y/n): ")
    if clobber.lower() != 'y':
      print(f"Skipping {url}")
      return
  print(f"Downloading {url} to {save_path}")
  response = requests.get(url, stream=True)
  if response.status_code == 200:
    with open(save_path, 'wb') as f:
      for chunk in response.iter_content(1024):
        f.write(chunk)
    print(f"Downloaded {url} to {save_path}")
  else:
    print(f"Failed to download {url}")

def main():
  os.makedirs('s3/models', exist_ok=True)
  for model in MODELS.items():
    if model is not None:
      try:
        download_model(model[1]["url"], model[1]["path"])
      except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
  main()