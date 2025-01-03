# Progetto Tesi: Implementazione di un'Architettura RAG per Assistenti Virtuali Universitari

Questo progetto implementa un'architettura **Retrieval-Augmented Generation (RAG)** per creare un assistente virtuale universitario. L'assistente combina modelli di linguaggio di grandi dimensioni (LLM) e database vettoriali per fornire risposte accurate e personalizzate alle domande degli studenti.

## Requisiti

1. **Python** 3.9 o superiore
2. **pip** per la gestione dei pacchetti
3. Librerie Python principali:
   - `sentence-transformers`
   - `chromadb`
   - `fastapi`
   - `uvicorn`
   - `openai`
4. Modello **MPNet multilingue**: `paraphrase-multilingual-mpnet-base-v2`
5. Accesso a **OpenAI API** (__inserire la api-key__).

## Configurazione dell'Ambiente

1. **Clona il Repository**
   ```bash
   git clone https://github.com/paoloscarpino/tesi-scarpino.git
   cd tesi-scarpino
   ```

2. **Crea un ambiente virtuale**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: .\venv\Scripts\activate
   ```

3. **Installa le dipendenze**
   ```bash
   pip install -r requirements.txt
   ```

4. **Imposta le Variabili di Ambiente Configura le chiavi necessarie per l'API di OpenAI**
   ```bash
   export OPENAI_API_KEY=tuo-api-key
   ```

## Popolamento del Database Vettoriale
1. **Preprocessa i Dati** Assicurati che i documenti della knowledge base siano preprocessati utilizzando il codice fornito nel progetto.
2. **Crea gli Embedding e Popola ChromaDB**:
   ```bash
   python populate_chromadb.py 
   ```

## Avvio del Backend
Esegui il server FastAPI per il chatbot:
   ```bash
   uvicorn main:app --reload
   ```

Il server sarà accessibile all'indirizzo: http://localhost:8000.

## Test in Locale

1. **Widget del Chatbot:** Apri il file widget.html nel browser e interagisci con il chatbot.
2. **Endpoint REST Utilizza strumenti come Postman o cURL per testare l'API:**
   ```bash
   curl -X POST "http://localhost:8000/chatbot" \
   -H "Content-Type: application/json" \
   -d '{"query": "Qual è la scadenza per le iscrizioni?"}'
   ```

## Struttura del Progetto
* main.py: Server FastAPI per gestire le query.
* generate_embeddings.py: Script per creare embeddings dei documenti.
* populate_chromadb.py: Script per popolare il database vettoriale.
* widget.html: Frontend del chatbot.
* requirements.txt: Elenco delle dipendenze.