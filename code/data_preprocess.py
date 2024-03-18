import os
import argparse
import re
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
import spacy
import random
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
random.seed(19999928)


def convert_format(d):
    """Convert invalid date format."""
    try:
        return f"{d[4:6]}/{d[7:9]}/{d[1:3]} {d[10:-1]}"
    except TypeError:
        return np.nan


def clean_dates(df, column):
    """Convert the date to datetime object."""
    df[column] = df[column].apply(convert_format)
    df[column] = pd.to_datetime(df[column], errors='coerce')


def prepare_datasets(filepath: str, nrows: int | None) -> tuple[pd.DataFrame]:
    """
    Reads post and question entry datasets from a specified filepath;
    merge question and post dataframe;
    cleans date columns.

    Args:
        filepath (str): The directory path where the CSV files are located.
        nrows (int | None): The number of rows to load from question dataset, positive integer.

    Returns:
        tuple: Merged posts DataFrame and attorney time DataFrame.

    Raises:
        FileNotFoundError: If any of the CSV files are not found at the specified filepath.
    """
    # Read data
    try:
        question = pd.read_csv(os.path.join(filepath, "questions.csv"), index_col=0)
        if nrows:
            question = question.sample(n=nrows)
    except FileNotFoundError as e:
        raise FileNotFoundError('question.csv not found')
    question = question[["StateAbbr", "QuestionUno", "Category", "TakenByAttorneyUno"]]
    question = question[question['TakenByAttorneyUno'].notna()]
    chunks = []
    try:
        for chunk in pd.read_csv(os.path.join(filepath, "questionposts.csv"), chunksize=10000, usecols=["Id", "QuestionUno", "PostText", "CreatedUtc"]):
            filtered_chunk = chunk.merge(question, on='QuestionUno', how='inner')
            chunks.append(filtered_chunk)
        post = pd.concat(chunks, ignore_index=True)
    except FileNotFoundError as e:
        raise FileNotFoundError('questionposts.csv not found')
    # Clean date
    post.PostText.fillna(" ", inplace=True)
    clean_dates(post, "CreatedUtc")
    return post




def preprocess_text(text_series: pd.Series) -> pd.Series:
    """
    Removes specific symbols and replaces URLs with the string 'url' from the text in the pandas Series.

    Args:
    - text_series (pd.Series): A Series of text strings to preprocess.

    Returns:
    - pd.Series: A Series where each text string has specified symbols removed.
    """

    # Define a regular expression pattern for symbols to remove
    # Include all symbols in the square brackets that you want to remove
    symbols_to_remove = r'[^\w\s]'
    url_pattern = r'https?://\S+|www\.\S+'
    text_series = text_series.apply(lambda text: re.sub(url_pattern, ' url ', str(text)))
    text_series = text_series.apply(lambda text: re.sub(symbols_to_remove, '', str(text)))
    tokenized = []
    for doc in tqdm(nlp.pipe(text_series, n_process=-1), total=len(text_series)):
        tokens = [token.lemma_.lower() for token in doc if not token.is_space]
        sent = " ".join(tokens)
        tokenized.append(sent)
    return pd.Series(tokenized)



def main(args):
    input_path = args.input_path
    output_path = args.output_path
    nrows = args.nrows

    # Init gloable var
    # global ATT_WORDS, CLI_WORDS
    # ATT_WORDS, CLI_WORDS = initialize_word_sets(os.path.join(input_path, "att_label.csv"))

    # Prepare datasets
    print("Prepare Data\n")
    post= prepare_datasets(input_path, nrows=nrows)

    # Clean text
    print("Clean Data\n")
    post["CleanedText"] = preprocess_text(post["PostText"])

    # # Label posts
    # print("Label Posts\n")f
    # # post = label_posts(post, attorney_time)

    # Save new post DataFrame without index
    post = post[["Id", "QuestionUno", "Category", "CleanedText"]]
    post.to_csv(os.path.join(output_path, "cleaned_posts.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and label post data.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input directory containing the data files.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output directory where the processed files will be saved.')
    parser.add_argument('-n', '--nrows', type=int, default=None, help='Optional: Number of rows to read from question dataset.')

    args = parser.parse_args()
    main(args)

