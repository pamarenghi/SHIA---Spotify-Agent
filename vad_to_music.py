import pandas as pd
import numpy as np
import ast
import json
import random

output_prompt = {
    'valence': 1.5,
    'arousal': 1.0,
    'dominance': 1.5
}

def vad_to_music(output_prompt):
    dict = {}
    k = 1000

    path = "Data/muse_v3.csv"
    df_main = pd.read_csv(path)

    df_main['seeds'] = df_main['seeds'].apply(ast.literal_eval)

    column_renaming = {'valence_tags': 'valence', 'arousal_tags': 'arousal', 'dominance_tags': 'dominance'}
    df_main = df_main.rename(columns=column_renaming)

    # Fonction pour calculer la distance euclidienne
    def compute_similarity(row, prompt, metric='cosine'):
        if metric == 'euclidean':
            return np.sqrt((row['valence'] - prompt['valence'])**2 + 
                        (row['arousal'] - prompt['arousal'])**2 + 
                        (row['dominance'] - prompt['dominance'])**2)
        
        elif metric == 'cosine':
            vector1 = np.array([row['valence'], row['arousal'], row['dominance']])
            vector2 = np.array([prompt['valence'], prompt['arousal'], prompt['dominance']])
            
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)

        else:
            raise ValueError("Métrique non supportée, choisissez 'euclidean' ou 'cosine'.")

    # Appliquer la fonction pour calculer la similarité
    df_main['similarity'] = df_main.apply(lambda row: compute_similarity(row, output_prompt, metric="euclidean"), axis=1)

    # Trier par similarité croissante et sélectionner les 5 musiques les plus proches
    df_sorted = df_main.sort_values(by='similarity', ascending=False).head(k)
    df_sorted = df_sorted.dropna(subset=['spotify_id'])
    counter = random.randint(0,50)
    # Afficher les résultats
    # print(df_sorted[['artist', 'similarity', 'spotify_id', 'track']])
    # print("ID Spotify du premier trouvé :", df_sorted.iloc[0]['spotify_id'])
    print(df_sorted.head(10))
    link = 'https://open.spotify.com/intl-fr/track/' + str(df_sorted.iloc[counter]['spotify_id'])
    # print(link)
    # print(dict)
    return link

link = vad_to_music(output_prompt)
print(link)