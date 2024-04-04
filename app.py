import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
import pickle,os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


with st.sidebar:
    st.title("LLM Chat App")
    st.markdown('''
    ##About 
    This app is an LLM-operated chatbot build using
    Langchain and OpenAI
    ''')
    add_vertical_space()
    st.write('Made by Anish Goyal')
    
    
def main():
    st.header('Chat with PDF')
    
    load_dotenv()
    pdf=st.file_uploader("Upload your PDF",type='pdf')
    
    
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        #st.write(pdf_reader)
        
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
        chunks=text_splitter.split_text(text)
        #st.write(chunks)
        store_name=pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorStore=pickle.load(f)
            st.write("Embeddings loaded from the disk")
        else:
            embeddings = OpenAIEmbeddings()
            vectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            st.write(vectorStore)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorStore,f)   
            st.write("Embeddings computations done")
        
        query=st.text_input("Ask question about your PDF Files")
        st.write(query)
        
        if query:
            docs=vectorStore.similarity_search(query,k=3)
            #st.write(docs)
            
            llm=OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain (llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
            
                response=chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)    
        
        
    
    
if __name__=='__main__':
    main()
    