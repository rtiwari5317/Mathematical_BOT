import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from utils.loadVectorStore import load_vectorstore
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='app.env')


print("Starting Telegram Bot...", os.getenv("GROQ_API_KEY"))

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv("GROQ_API_KEY"))
# llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=st.secrets["groq_api_key"])

vectorstore_path = 'C:\\Users\\HP\\Downloads\\ML-Projects-master\\ML-Projects-master\\8-MathsGPT\\Maths_Datasets\\pdf_vectorstore'
vectorstore = load_vectorstore(vectorstore_path, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I'm your MathBot. Ask me anything from your PDFs?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    answer = qa_chain.run(query)
    await update.message.reply_text(answer)

def main():
    # app = ApplicationBuilder().token(st.secrets["TELEGRAM_BOT_TOKEN"]).build()
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()



if __name__ == "__main__":
    main()