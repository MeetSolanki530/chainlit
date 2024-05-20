import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from chainlit import on_chat_start, on_message
from chainlit.message import Message
import chainlit as cl
from langchain_groq import ChatGroq
import json

template = """ Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatGroq(groq_api_key = "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")

# Instantiate the chain for that user session
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

history_file = "conversation_history.json"

@on_chat_start
def main():
    global history

    # Load history from the file if it exists
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []

@on_message
async def handle_message(message: Message):
    try:
        # Extract the plain text question from the Message object
        question = message.content

        # Add the question to the history
        history.append({"role": "user", "content": question})

        # Call the chain with the extracted question
        res = await llm_chain.acall(question)

        # Add the assistant's response to the history
        history.append({"role": "assistant", "content": res["text"]})

        # Save the history to the file
        with open(history_file, "w") as f:
            json.dump(history, f)

        # Do any post-processing here (optional)

        # Send the response
        await Message(content=res["text"]).send()
    except Exception as e:
        # Handle the error gracefully
        await Message(content="An error occurred. Please try again later.").send()

async def handle_ui_message(message: dict):
    if message["type"] == "get_history":
        await Message(content=json.dumps(history)).send()
