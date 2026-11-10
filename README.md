# Development of a PDF-Based Question-Answering Chatbot Using LangChain

## AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

## PROBLEM STATEMENT:
Searching through large PDFs manually is slow and inefficient. Traditional keyword searching fails to understand the actual meaning of questions. This project solves this by creating a chatbot pipeline that reads a PDF, saves its meaning in a database, and uses an AI model to answer user questions instantly and accurately.

## DESIGN STEPS:

### STEP 1:
**Load and Split PDF Documents:** Use `PyPDFLoader` to read all the text pages from the PDF file. Then, use `RecursiveCharacterTextSplitter` to cut the long text into small, clean chunks so the AI model can read it easily.

### STEP 2:
**Create Vector Embeddings and Database:** Pass the text chunks into `OpenAIEmbeddings` to convert the words into math vectors. Save these vectors into a `Chroma` database so the chatbot can perform fast semantic searches.

### STEP 3:
**Setup RetrievalQA Chain:** Initialize the large language model (`ChatOpenAI`) and link it to the vector database using LangChain's `RetrievalQA` module. When a user asks a question, the system pulls the correct matching text and uses the LLM to generate a natural answer.

## PROGRAM:
```python
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Load the API credentials
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# 2. Connect to the processed PDF vector store
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# 3. Setup the AI model engine
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 4. Build the Question-Answering pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

# 5. Ask a question from the PDF document
question = "What is the prerequisite for the class?"
result = qa_chain({"query": question})

# 6. Display the final answer
print("Question:", question)
print("Answer:", result["result"])
```

## OUTPUT:
<img width="1920" height="1080" alt="Screenshot (3175)" src="https://github.com/user-attachments/assets/451902ff-92f2-4cfa-9751-d91d7363781f" />


## RESULT:
The PDF-based question-answering chatbot was designed and implemented successfully using LangChain. The program accurately extracted text from the document, generated vector indices, and answered content-specific user queries correctly without any errors.
The PDF-based question-answering chatbot was designed and implemented successfully using LangChain. The program accurately extracted text from the document, generated vector indices, and answered content-specific user queries correctly without any errors.
