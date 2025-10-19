import os
import sys
from threading import Thread
from typing import Iterable
from huggingface_hub import snapshot_download

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# --- Theme and CSS Definition ---

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",  # SteelBlue base color
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

# --- Fix for Dots.OCR Processor Loading ---

# Define a local directory to cache the model
CACHE_PATH = "./model_cache"
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# Download the model files locally
model_path_d_local = snapshot_download(
    repo_id='rednote-hilab/dots.ocr',
    local_dir=os.path.join(CACHE_PATH, 'dots.ocr'),
    max_workers=20,
    local_dir_use_symlinks=False
)

# Modify the configuration file to fix the processor loading issue
config_file_path = os.path.join(model_path_d_local, "configuration_dots.py")

if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as f:
        input_code = f.read()

    lines = input_code.splitlines()
    if "class DotsVLProcessor" in input_code and not any("attributes = " in line for line in lines):
        output_lines = []
        for line in lines:
            output_lines.append(line)
            if line.strip().startswith("class DotsVLProcessor"):
                # Insert the attributes line to specify which processors to load
                output_lines.append("    attributes = [\"image_processor\", \"tokenizer\"]")

        # Write the modified content back to the file
        with open(config_file_path, 'w') as f:
            f.write('\n'.join(output_lines))
        print("Patched configuration_dots.py successfully.")

# Add the local model path to sys.path so transformers can use the modified code
sys.path.append(model_path_d_local)


# --- Model Loading ---

# Constants for text generation
MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Dots.OCR from the local, patched directory
MODEL_PATH_D = model_path_d_local
processor = AutoProcessor.from_pretrained(MODEL_PATH_D, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_D,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()


@spaces.GPU
def generate_image(text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """Generate responses for image input using the Dots.OCR model."""

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    images = [image.convert("RGB")]

    # Create the prompt for Dots.OCR
    messages = [
        {"role": "user", "content": [{"type": "image"}] + [{"type": "text", "text": text}]}
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": True
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text.replace("<|im_end|>", "").replace("<end_of_utterance>", "")
        yield buffer, buffer

# Define examples for image inference
image_examples = [
    ["Reconstruct the doc [table] as it is.", "https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR2/resolve/main/images/0.png"],
    ["OCR the image.....", "https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR2/resolve/main/images/1.png"],
    ["OCR the image.....", "https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR2/resolve/main/images/3.png"],
]

# Create the Gradio Interface
with gr.Blocks(css=css, theme=steel_blue_theme) as demo:
    gr.Markdown("# **Dots.OCR**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
            image_upload = gr.Image(type="pil", label="Upload Image", height=320)
            image_submit = gr.Button("Submit", variant="primary")
            gr.Examples(examples=image_examples, inputs=[image_query, image_upload])

            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)

        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            raw_output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=13, show_copy_button=True)
            with gr.Accordion("Formatted Result", open=True):
                formatted_output = gr.Markdown(label="Formatted Result")

    image_submit.click(
        fn=generate_image,
        inputs=[image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[raw_output, formatted_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(show_error=True)
