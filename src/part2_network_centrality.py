'''
PART 2: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Build a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is inline with the standards we're using in this class 
'''
from datetime import datetime
import os
import numpy as np
import pandas as pd
import networkx as nx
import json

# Build the graph
g = nx.Graph()

# Set up your dataframe(s) -> the df that's output to a CSV should include at least the columns 'left_actor_name', '<->', 'right_actor_name'

with open(r"C:\Users\swagm\problem-sets\problem-set-4\data\imdb_movies_raw.json",
          "r", encoding="utf-8") as in_file:
    # Don't forget to comment your code
    for line in in_file:
        # Don't forget to include docstrings for all functions

        # Load the movie from this line
        this_movie = json.loads(line)
            
        # Create a node for every actor
        for actor_id,actor_name in this_movie['actors']:
        # add the actor to the graph
            g.add_node(actor_id, name=actor_name)
    
        # Iterate through the list of actors, generating all pairs
        ## Starting with the first actor in the list, generate pairs with all subsequent actors
        ## then continue to second actor in the list and repeat
        
        i = 0 #counter
        for left_actor_id,left_actor_name in this_movie['actors']:
            for right_actor_id,right_actor_name in this_movie['actors'][i+1:]:

                # Get the current weight, if it exists
                current_w = g.get_edge_data(left_actor_id, right_actor_id, default={}).get('weight', 0)
                # Add an edge for these actors
                if left_actor_id != right_actor_id:
                    g.add_edge(left_actor_id, right_actor_id, weight=current_w + 1)
            i += 1
                
                


# Print the info below
print("Nodes:", len(g.nodes))

#Print the 10 the most central nodes
deg_centrality = nx.degree_centrality(g)
top_10_nodes = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 by degree centrality:")
for node_id, score in top_10_nodes:
    name = g.nodes[node_id].get('name', node_id)
    print(f"{name}\t{score:.5f}")

# Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`

rows = [{
    'left_actor_name': g.nodes[u].get('name', u),
    '<->': '<->',
    'right_actor_name': g.nodes[v].get('name', v),
} for u, v in g.edges()]

df_out = pd.DataFrame(rows)

fname = f"network_centrality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
out_path = os.path.join(os.path.dirname(__file__), '..', 'data', fname)
df_out.to_csv(out_path, index=False)