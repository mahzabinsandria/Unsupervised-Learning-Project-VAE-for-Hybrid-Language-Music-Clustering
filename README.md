# Unsupervised Music Clustering (VAE)

Repo layout follows the provided template:

project/

  data/audio/
  
  data/lyrics/
  
  notebooks/exploratory.ipynb
  
  src/{vae.py,dataset.py,clustering.py,evaluation.py}
  
  results/latent_visualization/
  
  results/clustering_metrics.csv
  

## Setup
pip install -r requirements.txt

If MP3 audio fails to load, install ffmpeg:
- Colab/Ubuntu: apt-get install ffmpeg

## Data (same datasets used in the implementation)
- Bangla lyrics CSV from GitHub (Hamza029/Bangla-Songs-Dataset)
- English lyrics from HuggingFace: brunokreiner/genius-lyrics
- Audio+lyrics from HuggingFace: jamendolyrics/jamendolyrics

Build the required CSVs and copy audio locally:
python src/dataset.py --prepare

This creates:
- data/lyrics/tracks_easy.csv  (id, lyrics, language)
- data/lyrics/tracks_av.csv    (id, audio_path, lyrics, language, genre)
and copies audio into data/audio/jamendolyrics/

AUDIO NOT INCLUDED IN GITHUB (too large)

Google Drive folder link:
https://drive.google.com/drive/folders/1uGaaCTVinSWNjrOC2g6QxxIe3dYLIcbW

How to use:
1) Mount Google Drive in Colab
2) Copy the Drive audio folder into: data/audio/
   
## 2) Datasets Used (Sources)

### Easy (Lyrics-only)
- Bangla lyrics CSV (GitHub):  
  https://github.com/Hamza029/Bangla-Songs-Dataset
- English lyrics (Hugging Face):  
  https://huggingface.co/datasets/brunokreiner/genius-lyrics

### Medium + Hard (Audio + Lyrics)
- JamendoLyrics (Hugging Face):  
  https://huggingface.co/datasets/jamendolyrics/jamendolyrics

---

## Run
Easy:
python src/vae.py --task easy --k 8

Medium:
python src/vae.py --task medium --k 6

Hard:
python src/vae.py --task hard --k 6 --beta 4.0

All:
python src/vae.py --task all --beta 4.0

Outputs:
- results/clustering_metrics.csv
- results/latent_visualization/*.png
