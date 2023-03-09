import openai

def getContext(query, index, EMBED_MODEL, OPENAI_API):
    
    openai.api_key = OPENAI_API
    
    res = openai.Embedding.create(
        input=[query],
        engine=EMBED_MODEL
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)

    pinecone_result = []
    
    # contexts = [x['metadata']['text'] for x in res['matches']]
    for x in res['matches']:
        pinecone_result.append({
            'title': x['metadata']['title'],
            'text': x['metadata']['text'],
            'url': x['metadata']['link'],
            'score': x['score']
                })
    
    return pinecone_result