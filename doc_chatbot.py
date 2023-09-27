import gradio as gr
import os
import pandas as pd
import requests
import textract
import openai
import gradio as gr
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
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = '******'  # OpenAIから取得したキー
folder_path = '******'  # フォルダ内の全てのPDFファイルをロードするフォルダパス
all_text = []  # ロードされたテキストを格納するためのリスト
chat_history = []

# フォルダ内のPDFファイルを一つずつ処理
for root, dirs, files in os.walk(folder_path):
  for file in files:
    if file.endswith(".pdf"):
      pdf_path = os.path.join(root, file)  # PDFファイルのパス
      doc = textract.process(pdf_path)  # PDFをテキストに変換
      text = doc.decode('utf-8')
      all_text.append(text)  # テキストをリストに追加
combined_text = "\n".join(all_text)  # テキストを結合

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24)

chunks = text_splitter.create_documents([combined_text])

embeddings = OpenAIEmbeddings()  # 埋め込みモデルの取得
db = FAISS.from_documents(chunks, embeddings)  # vector databaseの作成

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5), db.as_retriever())

def generate_answer(query):
    # qa関数に質問を送信し回答を取得
    result = qa({"question": query, "chat_history": chat_history})
    # 質問と回答をchat_historyに追加
    chat_history.append((query, result['answer']))
    return result['answer']

# Gradio UIを作成
ui = gr.Interface(
    fn=generate_answer,
    inputs=gr.inputs.Textbox(lines=1, label="質問を入力してください"),
    outputs=gr.outputs.Textbox(label="答え"),
    live=False,
    title="ISDLGPT",#文字の大きさ
    description="""
    <h1 style="font-size: 24px;">ISDL内のことならなんでもお聞きください。</h1>
    <p style="font-size: 18px;">役割担当や研究など、様々な用途でお使いいただけます。</p>
    """,
    css=".gradio-container {background-color: #f0f8ff}",
    article=img_html  # 画像のHTMLコードをarticleとして指定
)

# Gradio UIを表示
ui.launch(share=True)
