#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


pip install -U pip


# In[14]:


get_ipython().system('pip install langchain')


# In[9]:


get_ipython().system('pip install openai')


# In[21]:


get_ipython().system('pip install PyPDF2')


# In[33]:


get_ipython().system('pip install faiss-cpu')


# In[23]:


get_ipython().system('pip install tiktoken')


# In[24]:


get_ipython().system('pip install "unstructured[all-docs]"')


# In[25]:


from langchain.embeddings.openai import OpenAIEmbeddings 


# In[26]:


from langchain.text_splitter import CharacterTextSplitter


# In[27]:


from langchain.vectorstores import FAISS


# In[52]:


import os
os.environ["OPENAI_API_KEY"] = "OPEN AI KEY"


# In[50]:


from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('info')
documents = loader.load()


# In[53]:


documents


# In[5]:


documents[:100]


# In[54]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20) 
docs = text_splitter.split_documents(documents)


# In[55]:


docs[0].page_content


# In[56]:


from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# In[15]:


pip install -U langchain-openai


# In[ ]:





# In[57]:


docsearch = FAISS.from_documents(docs, embeddings)


# In[58]:


from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# In[59]:


docsearch


# In[42]:


from langchain.chains.question_answering import load_qa_chain 
from langchain.llms import OpenAI


# In[60]:


chain = load_qa_chain(OpenAI(), chain_type="stuff")


# In[45]:


query = "What is football ?"
docs = docsearch.similarity_search(query) 
chain.run(input_documents=docs, question=query)


# In[46]:


query = "explain about football in detail?"
docs = docsearch.similarity_search(query) 
chain.run(input_documents=docs, question=query)


# In[47]:


query = "what is intel?"
docs = docsearch.similarity_search(query) 
chain.run(input_documents=docs, question=query)


# In[64]:


query = "rules of football?"
docs = docsearch.similarity_search(query) 
chain.run(input_documents=docs, question=query)


# In[61]:


query = "full name of ronaldo?"
docs = docsearch.similarity_search(query) 
chain.run(input_documents=docs, question=query)


# In[63]:


query = "details on ronaldo?"
docs = docsearch.similarity_search(query) 
chain.run(input_documents=docs, question=query)


# In[ ]:




