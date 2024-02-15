# Q&A blog bot with embeddings

# Introduction

The diagram below briefly outlines how Retrieval Augmented Generation (RAG) models work. In short, they essentially *retrieve* additional context to *augment* the response *generated* by a LLM. 

The diagram below describes how embeddings are used to compare a prompt to a knowledge source in order to retrieve the most likely relevant context. The prompt and context is then provided to a LLM model (in this case gpt-3.5-turbo) to generate a contextually relevant response. 

![diagram](/screenshots/diagram.png)

## Diagram flow

1 : In the case of this particular implementation, the knowledge source is a blog. The knowledge is obtained by first extracting all the hyperlinks on the site, and discarding any that point to other domains. Each unique hyperlink is then visited and the content extracted into text files. The text files are then used to create a data frame. Each row in the data frame is tokenised, which  allows for analysing the length of documents, which is relevant for understanding the data's distribution and for optimising model input sizes. 

2 : After a bit more processing to create smaller chunks (if required), the embeddings are generated and saved; in this case, to a .csv file.

```bash
<SNIP>
https://emdeh.com/repositories
https://emdeh.com/news/announcement_7
https://emdeh.com/blog/2024/codify-walkthrough
Embeddings generated and saved to 'data/embeddings.csv'.
Preprocessing complete. Embeddings are ready.

# You can see the blog's links being iterated here.
```

3 - 5 :  When a user provides the prompt to the service it will also pass the prompt to the embeddings model to retrieve its vector.

![image of prompt](/screenshots/Pasted%20image%2020240215124006.png)

6: The service then compares the prompt's vector to the Vector DB (in this case, the .csv  file containing the blog's embeddings is loaded into another data frame). 

> *The comparision is done using Cosine function to calculate the distance between the question's embedding and each row's embedding in the data frame. Cosine distances is a measure used to determine the similarity between two vectors, with lower values indicating higher similarity.*

The service will then iterate over the data frame to accumulate the most similar text until it reaches a pre-defined token limit. This then forms the context for the original prompt.

7 - 9: The context, and original prompt, is then passed to the GPT model, which returns a generative completion. This completion is presented back to the end-user.

![image of completion](/screenshots/Pasted%20image%2020240215124051.png)

### Credits

The original code was adapted from **OpenAI's Web  Q&A with Embeddings tutorial**. Learn how to crawl your website and build a Q/A bot with the OpenAI API. You can find the full tutorial in the [OpenAI documentation](https://platform.openai.com/docs/tutorials/web-qa-embeddings).

## Structure
//TO-DO

## Detailed documentation
Follow the link to detailed code documentation

### [Continue to documentation...](/detailed-overview/1.%20Introduction.md)

## Using this project

### Clone or fork this repository

First you need to create your own version.

### Work from a virtual environment

To start, create a `venv` with `python3 -m venv venv` in the root directory.

Then run `source venv/bin/activate`; again from the `/root` directory.

This will install `requirements.txt` for you, which are primarily based on the [Q&A with embeddings tutorial](https://platform.openai.com/docs/tutorials/web-qa-embeddings).

You can also use `pip freeze > requirements.txt` to periodically update the `/requirements.txt` file if you're developing.

### You need to create an `.env` file for the API key

Create a `.env` file in the `/chatbot-backend` directory.

This file should store your OpenAI API key. It should look like this:

```bash
OPENAI_API_KEY=key-here-with-not-quotes
```