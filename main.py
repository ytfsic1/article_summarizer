import os
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter, PageObject
from gpt4all import GPT4All
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit

# Define input and output directories
input_directory = "./downloads"
output_directory = "./processed"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Configuration for GPT4All model
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
MODEL_PATH = "D:\\gpt4all\\models"
DEVICE = "cuda"
CONTEXT_SIZE = 80000

# Initialize GPT4All model
model = GPT4All(MODEL_NAME, model_path=MODEL_PATH, device=DEVICE, n_ctx=CONTEXT_SIZE, verbose=True)

def split_into_chunks(text, max_tokens):
    """
    Split text into smaller chunks that fit within the model's context window.

    Parameters:
        text (str): The text to split.
        max_tokens (int): Maximum tokens per chunk.

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_tokens:  # +1 for the space
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_summary(content):
    """
    Generate a summary of the provided content using GPT4All.

    Parameters:
        content (str): Text content to summarize.

    Returns:
        str: Summary of the content.
    """
    with model.chat_session():
        response = model.generate(
            prompt=f"Summarize the following text:\n{content}",
            max_tokens=500
        )
    return response.strip()

def generate_final_summary(summaries):
    """
    Generate a cohesive final summary from multiple chunk summaries using GPT4All.

    Parameters:
        summaries (list): List of summaries to combine into a cohesive summary.

    Returns:
        str: Final cohesive summary.
    """
    combined_summaries = "\n\n".join(summaries)
    with model.chat_session():
        response = model.generate(
            prompt=f"Combine the following summaries into a cohesive summary:\n{combined_summaries}",
            max_tokens=10000
        )
    return response #.strip().replace("\n", "\n\n")


def add_text_to_pdf(writer, text, page_width=612, page_height=792, font_size=12):
    """
    Add text to the PDF writer, creating as many pages as needed and ensuring text wraps within the page width.

    Parameters:
        writer (PdfWriter): The PdfWriter object to add pages to.
        text (str): The text to add to the PDF.
        page_width (int): The width of the PDF page.
        page_height (int): The height of the PDF page.
        font_size (int): The font size for the text.

    Returns:
        None
    """

    # Initialize a buffer for creating the PDF pages
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    text_obj = pdf_canvas.beginText(40, page_height - 40)
    text_obj.setFont("Helvetica", font_size)
    line_height = font_size + 2  # Set line height based on font size

    # Split the text into paragraphs and then into lines that fit within the page width
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        if paragraph.strip():  # Skip empty paragraphs
            lines = simpleSplit(paragraph, "Helvetica", font_size, page_width - 80)
            for line in lines:
                text_obj.textLine(line)
                if text_obj.getY() < 40:  # Add a new page if text reaches the bottom
                    pdf_canvas.drawText(text_obj)
                    pdf_canvas.showPage()
                    text_obj = pdf_canvas.beginText(40, page_height - 40)
                    text_obj.setFont("Helvetica", font_size)

    # Finish drawing on the last page
    pdf_canvas.drawText(text_obj)
    pdf_canvas.save()

    # Add the generated PDF pages to the writer
    buffer.seek(0)
    temp_reader = PdfReader(buffer)
    for page in temp_reader.pages:
        writer.add_page(page)


def process_pdf(input_path, output_path):
    """
    Process a PDF file, generate summaries, and save the final cohesive summary to a new file.

    Parameters:
        input_path (str): Path to the input PDF file.
        output_path (str): Path to save the processed PDF file.

    Returns:
        None
    """
    reader = PdfReader(input_path)
    writer = PdfWriter()

    # Extract text content from all pages
    full_text = "".join(page.extract_text() for page in reader.pages)

    # Split text into chunks that fit the model's context window
    chunks = split_into_chunks(full_text, max_tokens=CONTEXT_SIZE)

    # Generate summaries for each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        summaries.append(generate_summary(chunk))

    # Generate a final cohesive summary
    final_summary = generate_final_summary(summaries)

    # Add the final summary as the first pages of the PDF
    add_text_to_pdf(writer, final_summary)

    # Add original pages
    for page in reader.pages:
        writer.add_page(page)

    # Write the processed content to the new file
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

# Iterate through all PDF files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".pdf"):
        input_path = os.path.join(input_directory, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_p.pdf"
        output_path = os.path.join(output_directory, output_filename)

        print(f"Processing: {filename} -> {output_filename}")

        # Process the PDF file
        process_pdf(input_path, output_path)

print("Processing complete. Processed files are saved in the '/processed' directory.")
