import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

def main():
    st.set_page_config(page_title="LangChain + Streamlit Demo", page_icon="🤖")
    st.title("🤖 LangChain + Streamlit Example")
    st.markdown(
        """
        This is a simple demo showing how to hook up LangChain with Streamlit.
        Type a question below, and the OpenAI model will attempt to answer it.
        """
    )

    # 1. Choose your LLM and temperature
    # llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo")  # Adjust temperature as you like
    llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_KEY)


    # 2. Define a simple prompt template
    #    We’ll ask the model to “answer as helpfully as possible.”
    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="You are a helpful assistant. User asks: \"{user_input}\". Provide a concise answer."
    )

    # 3. Create an LLMChain with the prompt and the chosen LLM
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # 4. Streamlit UI: a text input box and a button
    user_question = st.text_input("❓ What do you want to ask the model?")

    if st.button("Generate Response"):
        if not user_question:
            st.warning("Please type something before clicking Generate.")
        else:
            with st.spinner("⏳ Generating response..."):
                # 5. Run the chain
                response = chain.run(user_input=user_question)

            # 6. Display the result
            st.subheader("💡 Model’s Response")
            st.write(response)

if __name__ == "_main_":
    main()