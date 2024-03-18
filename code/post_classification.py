from openai import OpenAI
import pandas as pd
import chardet
from tqdm import tqdm
import os
import argparse

from config import API_key
def classify_post(post):
    """
    Classifies a given post as written by an attorney or a client using OpenAI's API.
    Maps 'Attorney' to 1 and 'Unclassified' (including any errors or 'Client') to 0.

    Args:
    - post (str): The post text to classify.

    Returns:
    - str: Classification result ('Attorney' or 'Client'), or an error message.
    """
    client = OpenAI(api_key = API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "The following is a classification question that determines whether a given text is written by an attorney or a client. Usually client ask a question regarding legal issue and explain the information about the case, and the attorney usually give advice, give possible solution, or ask for more information about the case.Examples: 'my child be currently place with my ex by # # # . he accuse i of neglect and drug use . i admit to light drug use and be currently on random # # # for # # # what will the court hearing be about , also # # # be tell i that my current boyfriend have to move out , what be the fast way i can get my child back without move out my boyfriend or continue to be monitor by.'(client) 'they be correct . most provider do not save the content of text message that be send or recieve by customer . there be no law that require this . they may have the time and date text be send , only if it be # # # ed . if you pay for each text message , you # # # probably have that information . if you pay a fee for unlimited text , it probably be not save . t - mobile might have tower location information for where a customer be when he or she be texte . usually the tower location information be available with a subpoena only , and be so technical you would need an expert witness to analyze the datum . the only practical way to get a copy of the content of a text message be to see if it be save on the phone that send or recieve it . do you have it save on your phone ? if so , take a screenshot and print it to preserve it . print copy can be enter into evidence in court . you could also try to subpoena the phone of the person who send it , but they may just delete it . good luck.'(attorney)"},
                {"role": "user", "content": f"Determine if the following post is written by an attorney or a client, and respond only with 'attorney' or 'client': \"{post}\""},
            ], ##give some context, what does client and attorney ususally do
            max_tokens=100, 
            n=1,
            stop=None,
            temperature=0.5, # set seed
        )

        result = response.choices[0].message.content.strip().lower()



        # Interpretation of the result, if needed
        if "attorney" in result.lower():
            return "Attorney"
        elif "client" in result.lower():
            return "Client"
        else:
            return "Unclassified"
    except Exception as e:
        return f"An error occurred: {str(e)}" ## save the result to the df in a seperate function. 看看vectorized能不能，一下处理多句话，能不能快点

def classify_dataframe(df, text_column):
    """
    Apply classify_post to each row in the DataFrame and append the result.

    Args:
    - df (pandas.DataFrame): DataFrame containing the text to classify.
    - text_column (str): Name of the column in df that contains the text.

    Returns:
    - pandas.DataFrame: DataFrame with the new column 'classification_result'.
    """
    tqdm.pandas()
    # Check if the text_column exists in the DataFrame
    if text_column not in df.columns:
        raise ValueError(f"Column {text_column} does not exist in DataFrame.")
        
    # Apply classify_post to the specified column and create a new column with the results
    df['classification_result'] = df[text_column].progress_apply(classify_post)
    return df



def main(args):

    input_path = args.input_path
    output_path = args.output_path

    input_file = os.path.join(input_path, 'cleaned_posts.csv')
    output_file = os.path.join(output_path, 'cleaned_posts.csv')

    chunk_size = 10000
    chunk_list = []  

    print("Loading and classifying data...")
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        tqdm.pandas() 
        
        classified_chunk = classify_dataframe(chunk, 'CleanedText')
        chunk_list.append(classified_chunk)
        
        output_file = os.path.join(output_path, f'classified_posts_{chunk.index.start}_{chunk.index.stop}.csv')
        classified_chunk.to_csv(output_file, index=False)
        print(f"Saved classified posts to {output_file}")

    print("Combining chunks into a single file...")
    combined_classified_posts = pd.concat(chunk_list, ignore_index=True)
    combined_output_file = os.path.join(output_path, 'combined_classified_posts.csv')
    combined_classified_posts.to_csv(combined_output_file, index=False)
    print(f"Saved combined classified posts to {combined_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify post data.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input directory containing the data files.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output directory where the classified files will be saved.')
    args = parser.parse_args()
    main(args)

