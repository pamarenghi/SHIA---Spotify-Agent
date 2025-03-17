import pandas as pd
import numpy as np

output_prompt = {
    'valence': 4.5,
    'arousal': 3.0,
    'dominance': 6.5
}

path = "Data/main_dataset.csv"
df_main = pd.read_csv(path)

column_renaming = {'valence_tags': 'valence', 'arousal_tags': 'arousal', 'dominance_tags': 'dominance'}
df_main = df_main.rename(columns=column_renaming)

# Fonction pour calculer la distance euclidienne
def compute_similarity(row, prompt, metric='euclidean'):
    if metric == 'euclidean':
        # Calcul de la distance euclidienne
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
df_main['similarity'] = df_main.apply(lambda row: compute_similarity(row, output_prompt, metric="cosine"), axis=1)

# Trier par similarité croissante et sélectionner les 5 musiques les plus proches
df_sorted = df_main.sort_values(by='similarity').head(5)

# Afficher les résultats
print(df_sorted[['artist', 'song', 'similarity']])