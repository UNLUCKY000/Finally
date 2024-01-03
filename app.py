import streamlit as st
from transformers import pipeline
from einops import rearrange

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

def main():
    st.title("Chatbot")
    st.write("This is a chatbot powered by HuggingFace Transformers.")
    st.write("You can chat with the chatbot by entering text in the input box below.")

    messages = []

    while True:
        user_input = st.text_input("Your message:")
        if user_input:
            messages.append({
                "role": "user",
                "content": user_input,
            })
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            chatbot_response = outputs[0]["generated_text"]
            messages.append({
                "role": "chatbot",
                "content": chatbot_response,
            })
            st.write(chatbot_response)

if __name__ == "__main__":
    main()
