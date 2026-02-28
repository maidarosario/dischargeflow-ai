import pandas as pd
import numpy as np
from openai import OpenAI

client = OpenAI()

# Load diagnosis file
df = pd.read_csv("diagnosis.csv")

descriptions = df["LongDescription"].dropna().tolist()

print(f"Generating embeddings for {len(descriptions)} diagnoses...")

# Batch embedding (VERY IMPORTANT — much faster)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=descriptions
)

embeddings = [item.embedding for item in response.data]

# Save embeddings
np.save("diagnosis_embeddings.npy", embeddings)
df.to_csv("diagnosis_with_index.csv", index=False)

print("Embeddings saved successfully.")