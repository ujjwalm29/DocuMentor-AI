import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

from DocumentController import DocumentController
from generation.chat import Chat
from generation.claude_chat import ChatClaude
from generation.groq_chat import ChatGroq
from generation.openai_chat import ChatOpenAI

load_dotenv()
doc_controller = DocumentController()

# Check for correct access code
def check_access_code():
    code = st.sidebar.text_input("Enter your access code:")
    return code == os.getenv("USAGE_KEY")


# Setup the sidebar
def get_api_provider(api_provider):
    if api_provider == "Claude (Haiku)":
        return ChatClaude()
    elif api_provider == "OpenAI (GPT 3.5)":
        return ChatOpenAI()
    else:
        return ChatGroq()


async def create_chat_ui():
    st.sidebar.header("Menu")
    with st.sidebar:
        if st.button('Permanently delete all uploaded data'):
            doc_controller.delete_indexes()
        with st.form("upload-form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Upload PDF File", type=['pdf'], help="Upload here")
            submitted = st.form_submit_button("Upload File")
            if uploaded_file and submitted is not None:
                print(f"Working on file {uploaded_file.name}")
                file_location = f"data/pdf/{uploaded_file.name}"
                os.makedirs(os.path.dirname(file_location), exist_ok=True)
                with open(file_location, "wb+") as file_object:
                    file_object.write(uploaded_file.read())

                await doc_controller.process_text_and_store(file_location)
                uploaded_file = None
        api_provider = st.selectbox("Select API Provider", options=["OpenAI (GPT 3.5)", "Claude (Haiku)", "Groq (Llama 3 8B)"])


    st.title("PDF Question and Answering")
    st.text("Upload a PDF on the left and ask any questions about it!")
    st.text("By default a PDF about networks is uploaded.")
    st.text("Customize your LLM provider, query method, retrieval method etc on the left.")

    if prompt := st.chat_input("Ask a question?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context = doc_controller.search_and_retrieve_result(prompt)
            api_provider: Chat = get_api_provider(api_provider)
            st.write(f"{api_provider.get_final_generated_message(context=context, query=prompt)}")


def main():
    if check_access_code():
        asyncio.run(create_chat_ui())
    else:
        st.error("Please enter a valid access code to proceed.")


if __name__ == "__main__":
    main()
