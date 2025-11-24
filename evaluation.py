import numpy as np
import os
import sys
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    print(f"Error Import: {e}")
    sys.exit(1)

#Konfigurasi
INDEX_PATH = 'faiss_index_news'

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

#DataSet Pengujian
TEST_DATA = {
    "who is the 4th president in Indonesia?": ['doc_216097', 'doc_219818', 'doc_248031'],

    "What is the impact of Gudang Garam's layoffs according to Said Iqbal?": ['doc_205387', 'doc_242413', 'doc_186868'], 
    
    "earthquake in Maluku Utara": ['doc_192295', 'doc_219855', 'doc_218726']
}

#Rumus
def calculate_precision(retrieved_ids, relevant_ids):
    relevant_retrieved = [doc for doc in retrieved_ids if doc in relevant_ids]
    if not retrieved_ids: return 0.0
    return len(relevant_retrieved) / len(retrieved_ids)

def calculate_recall(retrieved_ids, relevant_ids):
    relevant_retrieved = [doc for doc in retrieved_ids if doc in relevant_ids]
    if not relevant_ids: return 0.0
    return len(relevant_retrieved) / len(relevant_ids)

def calculate_ndcg(retrieved_ids, relevant_ids, k=3):
    dcg = 0.0
    idcg = 0.0
    
    #Hitung DCG
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1 if doc_id in relevant_ids else 0
        dcg += rel / np.log2(i + 2)
        
    # 2. Hitung IDCG
    num_relevant = min(len(relevant_ids), k)
    for i in range(num_relevant):
        idcg += 1 / np.log2(i + 2)
        
    if idcg == 0: return 0.0
    return dcg / idcg

#Main
def main():
    print("--- MEMULAI EVALUASI RAG (RETRIEVAL) ---")
    
    if not os.path.exists(INDEX_PATH):
        print(f"Error: Index '{INDEX_PATH}' tidak ditemukan.")
        return
        
    print(f"1. Memuat Model Embedding ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("2. Memuat Index FAISS...")
    try:
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
    except Exception as e:
        print(f"Error load index: {e}")
        return
         
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    total_precision = 0
    total_recall = 0
    total_ndcg = 0
    count = 0

    print(f"\n3. Menguji {len(TEST_DATA)} Pertanyaan...\n")

    for query, true_ids in TEST_DATA.items():
        print(f"Q: {query}")
        
        # Lakukan Retrieval
        results = vectorstore.similarity_search(query, k=3)
        
        retrieved_ids = [doc.metadata.get('doc_id') for doc in results]
        
        # Hitung Skor
        p = calculate_precision(retrieved_ids, true_ids)
        r = calculate_recall(retrieved_ids, true_ids)
        n = calculate_ndcg(retrieved_ids, true_ids, k=3)
        
        print(f"   -> Retrieved: {retrieved_ids}")
        print(f"   -> Expected : {true_ids}")
        print(f"   -> Scores   : Precision={p:.2f}, Recall={r:.2f}, nDCG={n:.2f}")
        print("-" * 30)
        
        total_precision += p
        total_recall += r
        total_ndcg += n
        count += 1

    if count > 0:
        print("\n=== LAPORAN AKHIR EVALUASI ===")
        print(f"Rata-rata Precision : {total_precision/count:.4f}")
        print(f"Rata-rata Recall    : {total_recall/count:.4f}")
        print(f"Rata-rata nDCG      : {total_ndcg/count:.4f}")
        print("==============================")

if __name__ == "__main__":
    main()