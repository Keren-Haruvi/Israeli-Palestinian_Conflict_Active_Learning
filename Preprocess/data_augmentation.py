import ast
import time
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm

from Preprocess.pre_process import get_deberta_embedding

openai.api_key = "your_api_key"

def dropout_augmentation(embedding_list, threshold=0.1):
    """
    apply dropout to a list of embedding features by zeroing out some values based on a threshold.
    """
    embedding_array = np.array(embedding_list)
    dropout_mask = np.random.binomial(1, 1 - threshold, size=embedding_array.shape)
    return (embedding_array * dropout_mask).tolist()


def generate_argument(sentence, retries=2):
    """
    takes a sentence presenting an argument and generates:
    - rephrased version that maintains the same main idea
    - a shorter version capturing only the main idea 
    """
    query = """
            You are tasked with augmenting textual data that presents arguments related to the Israel-Palestine conflict. Given an argument sentence, please provide the following:

            1. A rephrased version of the sentence that maintains the same main idea but is articulated differently.
            2. A shorter version of the argument, capturing only the main idea.

            Your output should be structured as a dictionary with the following keys:
            - 'rephrase': [Result of the first task]
            - 'shorter': [Result of the second task]

            Example:

            Original Sentence: 'If you believe there is not a clear good side or bad side, consider visiting. This is especially relevant if you are a woman or part of the LGBT community. Do not forget to mention your nationality and religion casually.'

            {
            "rephrase": "If you think there are not clear distinctions between good and bad sides, you might want to visit the area. This is particularly relevant if you identify as a woman or as part of the LGBT community. Just keep in mind to casually mention your nationality and religion.",
            "shorter": "Visiting the region is essential for understanding the clear distinctions in the conflict, especially for women and LGBT individuals."
            }
            """
    for _ in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": query},
                    {"role": "user", "content": sentence}
                ],
            )
            result = response['choices'][0]['message']['content']
            return result
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(5)  # Wait a bit before retrying
    return 0  # Default to 0 if all retries fail


def add_sentences(df, text_col):
    """
    classify each sentence in the specified column and add a new column with the results
    """
    results = []
    for sentence in tqdm(df[text_col], desc="generate argument"):
        results.append(generate_argument(sentence))
    return results


def generate_GPT_arguments(df):
    """
    generate arguments using GPT-3.5 for each sentence in the df, 
    including rephrased and shorter texts, and their respective embeddings
    """
    result = add_sentences(df, 'text')
    result = [ast.literal_eval(res) for res in result]
    df['GPT3.5_rephrase_text'] = [res['rephrase'] for res in result]
    df['GPT3.5_shorter_text'] = [res['shorter'] for res in result]

    df['GPT3.5_rephrase_text_embedding'] = df['GPT3.5_rephrase_text'].apply(lambda text: get_deberta_embedding(text))
    df['GPT3.5_shorter_text_embedding'] = df['GPT3.5_shorter_text'].apply(lambda text: get_deberta_embedding(text))
    return df


if __name__ == "__main__":
    df = pd.read_csv('/home/student/Project/Datasets/processed_df.csv')
    df = generate_GPT_arguments(df)
    df.to_csv('/home/student/Project/Datasets/processed_df.csv', index=False)