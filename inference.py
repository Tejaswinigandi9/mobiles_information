import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import pipeline



# Constants
CSV_DIR = "./mobiles_data/brands"
OUTPUT_CSV = "./final.csv"
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def merge_csv_files(csv_dir, output_csv):
    """Merges all CSV files in the given directory into a single CSV file."""
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged {len(csv_files)} CSV files into {output_csv}")


def load_and_split_csv(file_path):
    """Loads CSV file and splits text into chunks."""
    loader = CSVLoader(file_path=file_path)
    documents = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks, persist_directory):
    """Saves text chunks into ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {persist_directory}.")
    return db


def query_chroma(query, top_k, persist_directory):
    """Searches ChromaDB for relevant documents."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    results = db.similarity_search(query, k=top_k)
    
    for i, doc in enumerate(results, start=1):
        print(f"Result {i}:\n{doc.page_content}\nMetadata: {doc.metadata}\n")
        
def infer_gemma(query):
    try:

        pipe = pipeline(
            "text-generation",
            access_token = "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",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",  # replace with "mps" to run on a Mac device
        )

        messages = [
            {"role": "user", "content": f"return yes if the {query} is about mobiles else no"},
        ]

        outputs = pipe(messages)
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
        print(assistant_response)
        return assistant_response
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Merge CSV files
    merge_csv_files(CSV_DIR, OUTPUT_CSV)
    
    # # Load, split, and save data to ChromaDB (Uncomment to run)
    chunks = load_and_split_csv(OUTPUT_CSV)
    save_to_chroma(chunks, CHROMA_PATH)
    
    # Query ChromaDB
    query = "Can you give details of samsung mobile?"
    infer_gemma(query)
    query_chroma(query, top_k=5, persist_directory=CHROMA_PATH)
