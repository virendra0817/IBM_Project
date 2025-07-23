import os
import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Translation function
def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
    )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated[0]

# Supported languages
languages = ["en", "hi", "fr", "de", "es", "zh", "ja", "ko"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 AI-Powered Multi-Lingual Translator (M2M100)")

    with gr.Row():
        input_text = gr.Textbox(label="Enter text to translate", lines=3)

    with gr.Row():
        src_lang = gr.Dropdown(choices=languages, value="en", label="Source Language")
        tgt_lang = gr.Dropdown(choices=languages, value="hi", label="Target Language")

    with gr.Row():
        translate_button = gr.Button("Translate")

    output_text = gr.Textbox(label="Translation Output", lines=3)

    translate_button.click(fn=translate, inputs=[input_text, src_lang, tgt_lang], outputs=output_text)

# Launch the app
demo.launch()
✅ requirements.txt (place in same repo)
nginx
Copy
Edit
transformers
sentencepiece
torch
gradio
📝 Notes:
This code is self-contained and compatible with Hugging Face Spaces

Does not require API tokens (because it loads a public model)

Should be placed in the root of your repo along with requirements.txt

✅ Folder structure (for GitHub or manual upload):
Copy
Edit
📁 IBM_Project/
├── app.py
├── requirements.txt
Then deploy to: https://huggingface.co/spaces

Would you like me to generate a sample README.md for your Hugging Face Space as well?









Ask ChatGPT

