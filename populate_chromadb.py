import os
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb

def leggi_e_preprocessa_pdf(pdf_path, output_csv):
    """
    Legge un file PDF, estrae il testo da ogni pagina e salva un dataset preprocessato in formato CSV.

    Args:
        pdf_path (str): Percorso al file PDF da processare.
        output_csv (str): Percorso del file CSV di output.
    """
    # Estrazione del nome del file
    nome_file = os.path.basename(pdf_path)
    nome_file_senza_estensione = os.path.splitext(nome_file)[0]

    pagine_info_array = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            raw_text = page.extract_text()

            pagine_info_array.append([nome_file, nome_file_senza_estensione, page_num + 1, raw_text])

    df_pdf = pd.DataFrame(pagine_info_array, columns=['DocName', 'Title', 'Page', 'Text'])
    df_pdf.to_csv(output_csv, index=False)
    print(f"Dataset preprocessato salvato in {output_csv}")

def genera_embeddings(dataset_csv, output_csv, modello):
    """
    Genera gli embeddings per il testo presente in un dataset CSV e salva il risultato.

    Args:
        dataset_csv (str): Percorso al file CSV contenente il testo preprocessato.
        output_csv (str): Percorso del file CSV di output con gli embeddings.
        modello (SentenceTransformer): Modello per la generazione degli embeddings.
    """
    df = pd.read_csv(dataset_csv)
    print("Generazione degli embeddings...")
    df['embedding'] = df['Text'].apply(lambda text: modello.encode(text).tolist())
    df.to_csv(output_csv, index=False)
    print(f"Embeddings salvati in {output_csv}")

def popola_chromadb(embeddings_csv, chroma_db_path, collection_name):
    """
    Popola un database ChromaDB con i documenti e gli embeddings generati.

    Args:
        embeddings_csv (str): Percorso al file CSV contenente il testo e gli embeddings.
        chroma_db_path (str): Percorso del database ChromaDB.
        collection_name (str): Nome della collezione da creare o aggiornare.
    """
    print("Configurazione di ChromaDB...")
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(collection_name)

    df = pd.read_csv(embeddings_csv)
    df['embedding'] = df['embedding'].apply(eval)  # Converti la stringa in lista

    print("Popolamento del database vettoriale...")
    for idx, row in df.iterrows():
        collection.add(
            documents=[row['Text']],
            embeddings=[row['embedding']],
            metadatas=[{
                'id_documento': f"{row['Title']}_{row['Page']}",
                'titolo': row['Title'],
                'pagina': row['Page']
            }],
            ids=[f"{row['Title']}_{row['Page']}"]
        )
    print("Database vettoriale popolato con successo!")

def main(pdf_path):
    """
    Esegue il flusso completo: preprocessa il PDF, genera gli embeddings e popola ChromaDB.

    Args:
        pdf_path (str): Percorso al file PDF di input.
    """
    dataset_csv = "dataset_preprocessato.csv"
    embeddings_csv = "embeddings_dataset.csv"
    chroma_db_path = "./chroma_db_mercatorum"
    collection_name = "assistent_virtuale_universita"

    # Preprocessa il PDF
    leggi_e_preprocessa_pdf(pdf_path, dataset_csv)

    # Genera gli embeddings
    modello = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    genera_embeddings(dataset_csv, embeddings_csv, modello)

    # Popola ChromaDB
    popola_chromadb(embeddings_csv, chroma_db_path, collection_name)

if __name__ == "__main__":
    pdf_path = input("Inserisci il percorso al file PDF: ").strip()
    main(pdf_path)
