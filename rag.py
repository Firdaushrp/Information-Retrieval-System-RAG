import torch
import sys
import os
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
except ImportError as e:
    print(f"Error Import: {e}")
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#Konfigurasi
INDEX_PATH = 'faiss_index_news'

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' 

LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

def load_llm_cpu():
    print(f"Sedang memuat model LLM: {LLM_MODEL} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        repetition_penalty=1.15,
        device=-1            
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def main():
    #Cek Folder Index
    if not os.path.exists(INDEX_PATH):
        print(f"ERROR: Folder '{INDEX_PATH}' tidak ditemukan.")
        print("TIPS: Jalankan 'python setup_index.py' dulu untuk membuat index.")
        return

    print("\n1. Memuat Database Berita (FAISS)...")
    # Load model embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    try:
        # Load Index
        vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embeddings
        )
    except Exception as e:
        print(f"Error load index: {e}")
        print("TIPS: Kemungkinan versi library beda atau model embedding beda.")
        print("SOLUSI: Jalankan ulang 'python setup_index.py' sekarang.")
        return
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    #Load LLM
    print("2. Memuat Otak AI (TinyLlama)...")
    try:
        llm = load_llm_cpu()
    except Exception as e:
        print(f"Error memuat LLM: {e}")
        return

    #Setup Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know. 
    
    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    #Rag Chain
    print("3. Menyiapkan Sistem RAG...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    print("\n" + "="*50)
    print(" SISTEM RAG SIAP! (Ketik 'exit' untuk keluar)")
    print("="*50)
    
    while True:
        # Input pertanyaan
        query = input("\nMasukan Pertanyaan (Bisa Indo/Inggris): ")
        if query.lower() in ['exit', 'quit', 'keluar']:
            break
        
        print(f"===> Mencari jawaban...")
        try:
            result = qa_chain.invoke({"query": query})
            
            raw_answer = result['result']
            if "Helpful Answer:" in raw_answer:
                answer = raw_answer.split("Helpful Answer:")[-1].strip()
            else:
                answer = raw_answer
                
            print(f"\n--- JAWABAN ---")
            print(answer)
            #<--
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()