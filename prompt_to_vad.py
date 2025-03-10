import pandas as pd
import string

PROMPT = "I am very angry!"

# Import
word_to_vad_path = "Data/word_to_VAD.csv"

df_word_to_vad = pd.read_csv(word_to_vad_path)

# List of words
def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

words = PROMPT.lower().split(' ')
cleaned_words = [remove_punctuation(word) for word in words]

# Mapping
V, A, D = 0, 0, 0
for word in cleaned_words:
    filter = df_word_to_vad['Word'] == word
    if df_word_to_vad[filter].shape[0] > 0:
        V += df_word_to_vad[filter]['Valence'].values[0]
        A += df_word_to_vad[filter]['Arousal'].values[0]
        D += df_word_to_vad[filter]['Dominance'].values[0]

V, A, D = V/len(cleaned_words), A/len(cleaned_words), D/len(cleaned_words)

results = {PROMPT: {'Valence': V, 'Arousal': A, 'Dominance': D}}

print(results)