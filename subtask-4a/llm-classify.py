import os
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
        f"Provide all applicable category numbers as a list (e.g., [1, 2], [2, 3], [1, 2, 3], [4]). If the tweet does not fit into any category, return [4].\n\n"
        f"Tweet: \"{tweet}\""
    )

    try:
        # Query the LLM
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant that classifies tweets."},
        #         {"role": "user", "content": prompt},
        #     ],
        #     temperature=0,  # Make output deterministic
        # )
        response = ollama.chat(model="deepseek-r1:14b", messages=[{"role": "system", "content": "You are a helpful assistant that classifies tweets into 0 or more categories."},
                                                         {"role": "user", "content": prompt}])
        
        # Extract the classification from the response
        classification = extract_answer_list(response['message']['content'])
        
        # Return the classification result
        return classification
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Sample tweet input
    # [1, 2]
    tweet = "A recent study published in Nature shows that AI can predict protein folding with high accuracy."
    category = classify_tweet(tweet)
    print(f"The tweet is classified as category: {category}")

    # [4] none
    tweet = "McDonald's breakfast stop then the gym"
    category = classify_tweet(tweet)
    print(f"The tweet is classified as category: {category}")

    # 1
    tweet = "65 percent of cats born with blue eyes are deaf."
    category = classify_tweet(tweet)
    print(f"The tweet is classified as category: {category}")

    # 2, 3
    tweet = "@ user Please read this research analysis https://www.apa.org/pubs/journals/releases/psp-pspp0000147.pdf"
    category = classify_tweet(tweet)
    print(f"The tweet is classified as category: {category}")

    # 3
    tweet = "How is University of Chicago shaping the future of science? Find out on April 6"
    category = classify_tweet(tweet)
    print(f"The tweet is classified as category: {category}")

    # 1, 2, 3
    tweet = "A fifth of US high school students use tobacco, finds survey http://www.bmj.com/content/349/bmj.g6885"
    category = classify_tweet(tweet)
    print(f"The tweet is classified as category: {category}")
