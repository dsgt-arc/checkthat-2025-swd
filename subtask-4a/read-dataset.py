import pandas as pd
import os
import time
import ollama
import re

from openai import OpenAI
from dotenv import load_dotenv, dotenv_values

def extract_answer_list(llm_output: str):
    """Extracts the final 3-element list from an LLM output."""
    match = re.search(r"\[\s*(\d+\.\d+,\s*\d+\.\d+,\s*\d+\.\d+)\s*\]", llm_output)
    if match:
        return [float(num) for num in match.group(1).split(",")]
    return []

# loading variables from .env file
load_dotenv() 

# Initialize OpenAI API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

#MODEL_NAME = "gpt-4"
MODEL_NAME = "deepseek-r1:14b"

def read_dataset(file_path):
    """
    Reads the dataset from the specified file path and returns it as a pandas DataFrame.
    """
    data = pd.read_csv(file_path, sep='\t') #.drop(columns=['index'])
    return data

def classify_tweet(tweet):
    """
    Classifies a tweet into one or more of the following categories:
    1. Contains a scientific claim.
    2. Refers to a scientific study/publication.
    3. Mentions scientific entities (e.g., a university or scientist).
    4. None of the above.
    If the tweet falls into multiple categories, all applicable categories are returned as a list.
    """
    # Define the prompt for the LLM
    prompt = (
        f"Classify the following tweet into one or more of the following categories:\n"
        f"1. Contains a scientific claim.\n"
        f"2. Refers to a scientific study/publication.\n"
        f"3. Mentions scientific entities (e.g., a university or scientist).\n"
        f"4. None of the above.\n"
        f"Provide all applicable category numbers as a one-hot encoded list of size 3 (e.g., [1.0, 1.0, 0], [1.0, 0, 0], [0.0, 0.0, 1.0]). If the tweet does not fit into any category, return [0.0, 0.0, 0.0]. You must always return a list of 3 elements as such.\n\n"
        f"Tweet: \"{tweet}\""
    )

    try:
        # Query the LLM
        # response = client.chat.completions.create(
        #     model=MODEL_NAME,
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant that classifies tweets."},
        #         {"role": "user", "content": prompt},
        #     ],
        #     temperature=0,  # Make output deterministic
        # )
        
        # # Extract the classification from the response
        # classification = response.choices[0].message.content

        response = ollama.chat(model=MODEL_NAME, 
                               messages=[{"role": "system", "content": "You are a helpful assistant that classifies tweets into 0 or more categories."},
                                         {"role": "user", "content": prompt}],
                               options={"temperature": 0.0})
        
        # Extract the classification from the response
        classification = extract_answer_list(response['message']['content'])
        
        # Return the classification result
        return classification
    except Exception as e:
        return f"Error: {str(e)}"
    

if __name__ == "__main__":
    data = "ct_train_data.tsv"

    # Read the dataset
    df = read_dataset(data)
    print(df.head())

    predictions = pd.DataFrame(columns=['index', 'prediction'])

    # Classify tweets
    for index, row in df.iterrows():
        category = classify_tweet(row['text'])
        predictions = pd.concat([predictions, pd.DataFrame({'index': [index], 'prediction': [category]})], ignore_index=True)
        #time.sleep(1)
        #print(f"Tweet: {row['text']}", f"Category: {category}")
        
        if index % 100 == 0:
            print(f"Processed {index} tweets.")

    print(predictions.head())

    # Write predictions to a new TSV file
    predictions.to_csv(MODEL_NAME + "-predictions.tsv", sep='\t', index=False)