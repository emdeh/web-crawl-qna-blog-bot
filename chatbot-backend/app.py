import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
from ast import literal_eval
import os
from dotenv import load_dotenv
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the embeddings DataFrame
df = pd.read_csv('data/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

# Access the OPENAI_API_KEY environment variable
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    #q_embeddings = client.embeddings.create(model="text-embedding-ada-002", input=question)['data'][0]['embedding']
    response = client.embeddings.create(input=[question], model="text-embedding-ada-002")
    q_embeddings = response.data[0].embedding

    # Calculate cosine distances and update the dataframe
    def calculate_cosine_distance(row_embedding):
        return cosine(q_embeddings, row_embedding)

    # Apply the function to each row's embeddings to calculate distances
    df['distances'] = df['embeddings'].apply(calculate_cosine_distance)

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    question,
    model="gpt-3.5-turbo",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
        print("\n\nQuestion:\n" + question)

    try:
        # Create a chat completion using the question and context
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Answer this question based on the context below: {question}. If the question can't be answered based on the context, say \"SALSA\"\n\n"},
                {"role": "user", "content": f"Context: {context}\n\n---\n\nAnswer:"},
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence)
        
        # Print the entire response object for inspection
        if debug:
            print("\n\nFull Response Object:\n")
            print(response)  # This line prints the entire response object

        # Extracting and returning the completion text
        completion_text = response.choices[0].message.content.strip()
        return completion_text
    
    except Exception as e:
        print(e)
        return ""

@app.route('/ask', methods=['POST'])
def ask():
    content = request.json
    question = content.get('question')

    answer = answer_question(df, question=question, debug=False)  # Assuming debug=True for development
    return jsonify({'answer': answer})

@app.route('/')
def home():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=False)
