# **dots.ocr-fix-demo**

This Gradio application demonstrates the capabilities of the "dots.ocr" model, a powerful multilingual document parser. It allows users to upload an image, provide a text query, and receive Optical Character Recognition (OCR) results. This particular demo also includes a crucial fix for a processor loading issue with the original model.

## About dots.ocr

"dots.ocr" is a state-of-the-art, 1.7B parameter vision-language model designed for parsing documents. It excels at:

*   **Multilingual Document Parsing:** Supporting over 100 languages.
*   **Unified Layout Detection and Content Recognition:** Handling text, tables, and mathematical formulas within a single model.
*   **Maintaining Reading Order:** Accurately preserving the structure of the original document.
*   **Efficient Performance:** Offering fast inference speeds due to its compact size.

## The Fix

A key feature of this application is the included patch for a known issue with the "dots.ocr" model's processor loading mechanism. The `configuration_dots.py` file is modified to explicitly define the `attributes` for the `DotsVLProcessor` class, ensuring that both the `image_processor` and `tokenizer` are loaded correctly. This prevents potential errors and allows the model to function as intended.

## How to Use the Application

The user interface, built with Gradio, is straightforward:

1.  **Query Input:** Enter a text prompt to guide the OCR process. Examples include "Reconstruct the doc [table] as it is." or "OCR the image.....".
2.  **Upload Image:** Select an image file from your local machine that you want to process.
3.  **Submit:** Click the "Submit" button to run the model.

The application provides both a raw output stream and a formatted result for easy viewing.

### Advanced Options

For more control over the text generation process, you can adjust the following parameters in the "Advanced options" section:

*   **Max new tokens:** The maximum number of tokens to be generated.
*   **Temperature:** Controls the randomness of the output. Higher values result in more creative but potentially less coherent text.
*   **Top-p (nucleus sampling):** A method for selecting the next token from a probability distribution.
*   **Top-k:**  Another method for token selection, limiting the choices to the 'k' most likely next tokens.
*   **Repetition penalty:**  Penalizes the model for repeating tokens, encouraging more diverse output.

## Code Overview

The application is built using Python and leverages several key libraries:

*   **Gradio:** For creating the interactive web interface.
*   **Hugging Face Transformers:** For loading and using the "dots.ocr" model and processor.
*   **PyTorch:** The deep learning framework on which the model is built.
*   **Pillow (PIL):** For image processing.

The code performs the following main functions:

1.  **Theme and CSS:** Defines a custom "SteelBlueTheme" for the Gradio interface.
2.  **Model Caching and Patching:** Downloads the "dots.ocr" model locally and applies the necessary fix to the configuration file.
3.  **Model Loading:** Loads the patched model and processor for use.
4.  **Image Generation Function:** Defines the core logic for processing user input, running the model, and streaming the output.
5.  **Gradio Interface:** Creates the user interface with input fields, buttons, and output displays.
