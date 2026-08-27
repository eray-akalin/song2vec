# Song2Vec: Playlist-Based Music Recommendation with Word2Vec

An implementation and empirical study of **Song2Vec**, applying Word2Vec (skip-gram with negative sampling) to playlist data for music recommendation. Songs that co-occur in user-created playlists are treated as "words in sentences," letting the model learn semantic similarity between tracks purely from co-listening patterns.

This project follows the hyperparameter analysis from:

> Caselles-Dupré, H., Lesaint, F., & Royo-Letelier, J. (2018). [*Word2vec applied to Recommendation: Hyperparameters Matter*](https://arxiv.org/abs/1804.04212). RecSys 2018.

which shows that NLP-default Word2Vec hyperparameters (small windows, `ns_exponent=0.75`) are suboptimal for recommendation tasks, and that re-tuning them for the co-occurrence structure of playlists significantly improves performance.

## Pipeline

```
Playlist data → Track sequences → Word2Vec training → Embedding-based recommendation & evaluation
```

1. **Parsing** — Extract `(artist, title)` track sequences from the [30Music dataset](http://recsys.deib.polimi.it/datasets/) (idomaar format).
2. **EDA** — Playlist length distribution, track frequency (long-tail), vocabulary statistics.
3. **Training** — Baseline skip-gram Word2Vec, then a systematic one-at-a-time hyperparameter sweep (`sg`, `window`, `negative`, `ns_exponent`, `min_count`, `vector_size`, `epochs`).
4. **Evaluation** — Hit Rate@K and NDCG@K (K = 5, 10, 20, 50) via next-song prediction, comparing context-averaging vs. single-query retrieval.
5. **Qualitative analysis** — Nearest-neighbor retrieval, vector arithmetic ("song algebra"), t-SNE visualization of the embedding space.
6. **Recommender + cold start** — A seed-based recommendation function with an artist-level fallback strategy for out-of-vocabulary songs.
7. **Failure analysis** — Categorized error cases with two proposed fixes (artist-diversity filtering, popularity-penalized re-ranking).

## Results

- Re-tuning hyperparameters for the recommendation setting (larger `window`, higher `negative`, adjusted `ns_exponent`) outperforms NLP-default settings.
- Single-query retrieval consistently beats context-averaging, especially on long playlists.
- Artist-level embedding fallback recovers a meaningful fraction of otherwise-unrecoverable OOV recommendations.

See [`song2vec.ipynb`](song2vec.ipynb) for the full analysis, plots, and numbers.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data

The dataset is not included in this repo (it's large and separately licensed). Download the **30Music** dataset in idomaar format and place these files under `data/`:

```
data/tracks.idomaar
data/playlist.idomaar
data/persons.idomaar
```

### Run

```bash
jupyter lab song2vec.ipynb
```

Running the notebook end-to-end regenerates `best_song2vec_model.model` and the figures in this repo.

## Repo structure

```
song2vec.ipynb              # Main notebook: full pipeline, analysis, and report
playlist_length_dist.png    # EDA figure
track_freq_dist.png         # EDA figure
data/                       # (gitignored) place 30Music idomaar files here
requirements.txt
```
