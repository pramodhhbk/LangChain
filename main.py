from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import configparser
config = configparser.ConfigParser()
config.read('config.cfg')
api_key = config.get('DEFAULT', 'api_key')


OPENAI_API_KEY = api_key

loader = CSVLoader(file_path="/content/Nike_US_Sales_Datasets.csv")
data = loader.load()


text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(data)


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()


prompt_template = ChatPromptTemplate.from_template(
    """
    You are an AI assistant specialized in marketing campaign strategy optimization. Your role is to assist marketers with queries related to:
    1. Designing effective marketing campaigns.
    2. Maximizing engagement and ROI within a $100,000 budget.
    3. Selecting optimal marketing channels such as social media, email, influencer marketing, and events.
    4. Leveraging historical campaign performance metrics to improve strategies.

    Use the provided dataset to answer questions accurately. Respond in a professional and insightful manner.

    Context:
    {context}

    Marketer: {input}
    """
)


chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)


qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model, retriever=retriever, chain_type="stuff", return_source_documents=True
)


def handle_query(query):
    result = qa_chain({"query": query})
    context = "\n".join([doc.page_content for doc in result['source_documents']])
    return f"Context:\n{context}\n\nAnswer:\n{result['result']}"


if __name__ == "__main__":
    print("Welcome to the Marketing Campaign Master Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! Good luck with your campaigns!")
            break
        response = handle_query(user_input)
        print("Bot:", response)
