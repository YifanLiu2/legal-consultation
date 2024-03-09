import openai

# Replace "your_api_key_here" with your actual OpenAI API key
openai.api_key = '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'

def classify_post(post):
    """
    Classifies a given post as written by an attorney or a client using OpenAI's API.

    Args:
    - post (str): The post text to classify.

    Returns:
    - str: Classification result ('Attorney' or 'Client'), or an error message.
    """
    try:
        response = openai.completions.create(
            engine="text-davinci-003",
            prompt=f"Determine if the following post is written by an attorney or a client: \"{post}\"", # give some examples, and the structure of the response
            max_tokens=60, 
            n=1,
            stop=None,
            temperature=0.5,
        )
        result = response.choices[0].text.strip()
        
        # Interpretation of the result, if needed
        if "attorney" in result.lower():
            return "Attorney"
        elif "client" in result.lower():
            return "Client"
        else:
            return "Unclassified"
    except Exception as e:
        return f"An error occurred: {str(e)}" ## save the result to the df in a seperate function. 看看vectorized能不能，一下处理多句话，能不能快点

# Example usage
posts = [
    "Can someone explain the legal implications of this contract clause?",
    "As per Section 2(b), the aforementioned stipulation lacks consideration, thus rendering it unenforceable."
]

for post in posts:
    result = classify_post(post)
    print(f"Post: {post}\nClassified as: {result}\n")
