from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# Caricamento del modello di embedding
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Configurazione di ChromaDB
client = chromadb.PersistentClient(path="./chroma_db_mercatorum")
collection = client.get_or_create_collection("assistent_virtuale_universita")

app = FastAPI()

clientAI = OpenAI()

# Configurazione CORS per permettere le richieste dal frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione, specificare il dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_query_embedding(query):
    embedding = embedding_model.encode([query]).tolist()
    return embedding[0]

def search_similar_documents(query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    documents = results['documents'][0]
    return documents

def build_context(relevant_docs):
    context = "\n".join(relevant_docs)
    return context

def generate_answer(query, context):
    prompt = f"\nRispondi alla seguente Domanda:\n{query}\n Usa il seguente contesto:\n{context}\n\nRisposta:"
    completion = clientAI.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": "Sei un super assistente vistuale che aiuta gli studenti a risolvere i loro problemi."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0,
                max_tokens=550,
            )
    
    answer = completion.choices[0].message.content
    
    return answer

@app.post("/chatbot")
async def chatbot_endpoint(request: Request):
    data = await request.json()
    query = data.get('query', '')
    if not query:
        return {"answer": "Per favore, inserisci una domanda valida."}
    query_embedding = get_query_embedding(query)
    relevant_docs = search_similar_documents(query_embedding)
    context = build_context(relevant_docs)
    answer = generate_answer(query, context)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
