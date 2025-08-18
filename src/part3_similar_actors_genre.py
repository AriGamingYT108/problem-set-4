'''
PART 2: SIMILAR ACTROS BY GENRE

Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''

#Write your code below

from datetime import datetime
from collections import defaultdict
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances

out_path = os.path.join(os.path.dirname(__file__), '..', 'data', f"similar_actors_genre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

query_actor_id = "nm1165110"

genre_set = set()
actor_name = {}  # actor_id -> actor_name
counts = defaultdict(lambda: defaultdict(int))  # counts[actor_id][genre] = frequency

with open(r"C:\Users\swagm\problem-sets\problem-set-4\data\imdb_movies_raw.json", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        m = json.loads(line)
        genres = m.get("genres", [])
        if not genres:
            continue
        genre_set.update(genres)

        # For every actor in the movie, increment each of the movie's genres once
        for aid, aname in m.get("actors", []):
            if aid is None or aname is None:
                continue
            actor_name.setdefault(aid, aname)
            for g in genres:
                counts[aid][g] += 1

# Create a matrix for the actors and their genre counts
actors = list(counts.keys())
genre_list = sorted(genre_set)
matrix = np.zeros((len(actors), len(genre_list)))

for i, actor_id in enumerate(actors):
    for j, genre in enumerate(genre_list):
        matrix[i, j] = counts[actor_id][genre]

df_matrix = pd.DataFrame(matrix, index=actors, columns=genre_list)
df_matrix.insert(0, "actor_name", [actor_name.get(aid, aid) for aid in actors])

if query_actor_id not in df_matrix.index:
    raise ValueError(f"Query actor {query_actor_id} not found in matrix.")

X = df_matrix[genre_list].to_numpy(dtype=float)
ids = df_matrix.index.to_numpy()

q_idx = int(np.where(ids == query_actor_id)[0][0])
q_vec = X[q_idx:q_idx+1]

cos_dist = cosine_distances(q_vec, X).ravel()

euc_metric = DistanceMetric.get_metric('euclidean')
euc_dist = euc_metric.pairwise(q_vec, X).ravel()

# exclude the actor from their own ranking
cos_dist[q_idx] = np.inf
euc_dist[q_idx] = np.inf

# Top-10 (smaller distance = more similar)
top10_cos_idx = np.argsort(cos_dist)[:10]
top10_euc_idx = np.argsort(euc_dist)[:10]

top10_cos = pd.DataFrame({
    "actor_id": ids[top10_cos_idx],
    "actor_name": df_matrix.iloc[top10_cos_idx]["actor_name"].values,
    "cosine_distance": cos_dist[top10_cos_idx]
})
top10_euc = pd.DataFrame({
    "actor_id": ids[top10_euc_idx],
    "actor_name": df_matrix.iloc[top10_euc_idx]["actor_name"].values,
    "euclidean_distance": euc_dist[top10_euc_idx]
})

# Save cosine results
top10_cos.to_csv(out_path, index=False)

# Print results + comparison
print("Top 10 similar by COSINE distance to Chris Hemsworth (nm1165110):")
for row in top10_cos.itertuples(index=False):
    print(f"{row.actor_id} {row.actor_name}")

# Describe how Euclidean changes the list
cos_set = set(top10_cos["actor_id"])
euc_set = set(top10_euc["actor_id"])
overlap = cos_set & euc_set
only_cos = cos_set - euc_set
only_euc = euc_set - cos_set


print(f"- Overlap: {len(overlap)} / 10")
if only_cos:
    print("- Only in COSINE:", ", ".join(f"{aid} {actor_name.get(aid, aid)}" for aid in only_cos))
if only_euc:
    print("- Only in EUCLIDEAN:", ", ".join(f"{aid} {actor_name.get(aid, aid)}" for aid in only_euc))

print("\nHow does the list change with EUCLIDEAN distance?")
print(
    "With EUCLIDEAN distance, the top-10 shifts toward more productive actors"
    "who have similar absolute numbers across genres. COSINE ignores magnitude and focuses on "
    "the pattern/proportions of genres, so it keeps actors with similar genre mix even if one "
    "has many more or fewer films."
)
print(f"Saved CSV -> {os.path.abspath(out_path)}")