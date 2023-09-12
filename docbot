import os
import pandas as pd
import requests
import textract
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# ここにOpenAIから取得したキーを設定します。
os.environ["OPENAI_API_KEY"] = '*****'

from google.colab import drive
drive.mount('/content/drive')

from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter

# フォルダ内の全てのPDFファイルをロードするフォルダパス
folder_path = "*****"

# ロードされたテキストを格納するためのリスト
all_text = []

# フォルダ内のPDFファイルを一つずつ処理
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".pdf"):
            # PDFファイルのパス
            pdf_path = os.path.join(root, file)

            # PDFをテキストに変換
            doc = textract.process(pdf_path)
            text = doc.decode('utf-8')

            # テキストをリストに追加
            all_text.append(text)

# テキストを結合
combined_text = "\n".join(all_text)

# GPT-2トークナイザーを初期化
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# トークン数をカウントする関数
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# テキストを分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([combined_text])

# Get embedding model
embeddings = OpenAIEmbeddings()
# vector databaseの作成
db = FAISS.from_documents(chunks, embeddings)

from IPython.display import display
import ipywidgets as widgets

# vextordbをretrieverとして使うconversation chainを作成。これはチャット履歴の管理も可能。
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=1.3), db.as_retriever())

chat_history = []

def on_submit(_):
    query = input_box.value
    input_box.value = ""

    if query.lower() == 'exit':
        print("Thank you for using the State of the Union chatbot!")
        return

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))

    display(widgets.HTML(f'<b>User:</b> {query}'))
    display(widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}'))

print("Welcome to the Transformers chatbot! Type 'exit' to stop.")

input_box = widgets.Text(placeholder='Please enter your question:')
input_box.on_submit(on_submit)

display(input_box)
