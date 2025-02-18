import os
from PyPDF2 import PdfReader, PdfWriter
from natsort import natsorted

def split_pdf_into_seven(input_path, output_dir, part_counter):
    """
    Splits a PDF into seven parts, renaming each part sequentially.

    Args:
        input_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output files.
        part_counter (int): Starting number for part naming.

    Returns:
        int: Updated part counter after processing the file.
    """
    print(f"Processing '{input_path}'...")

    # Read the input PDF
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")

    # Calculate the number of pages per part
    pages_per_part = total_pages // 20
    remaining_pages = total_pages % 20

    # Define the ranges for each part
    parts = []
    start = 0
    for i in range(20):
        end = start + pages_per_part + (1 if i < remaining_pages else 0)
        parts.append((start, end))
        start = end

    # Clean the base name of the file
    base_name = os.path.basename(input_path).replace(".pdf", "")
    base_name = "_".join(part for part in base_name.split("_") if not part.lower().startswith("part"))

    # Split and save each part
    for start, end in parts:
        writer = PdfWriter()
        for page in range(start, end):
            writer.add_page(reader.pages[page])

        output_path = os.path.join(output_dir, f"{base_name}_part{part_counter}.pdf")
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        print(f"Saved: {output_path}")
        part_counter += 1

    return part_counter

def normalize_file_name(file_name: str) -> str:
    """
    Normalize a file name by replacing spaces with underscores.
    Optionally, you can add more transformations like converting to lowercase.

    Args:
        file_name (str): The original file name.

    Returns:
        str: The normalized file name.
    """
    # Replace spaces with underscores
    normalized = file_name.replace(" ", "_")

    # Optionally: Add more transformations, e.g., convert to lowercase
    # normalized = normalized.lower()

    return normalized

def normalize_files_in_directory(directory_path: str):
    """
    Normalize all file names in the specified directory by replacing spaces with underscores.

    Args:
        directory_path (str): The path to the directory containing the files.

    Returns:
        None
    """
    for file_name in os.listdir(directory_path):
        # Skip directories
        if os.path.isdir(os.path.join(directory_path, file_name)):
            continue

        # Generate the normalized file name
        normalized_name = normalize_file_name(file_name)

        # Rename the file if the name has changed
        if file_name != normalized_name:
            original_path = os.path.join(directory_path, file_name)
            normalized_path = os.path.join(directory_path, normalized_name)
            os.rename(original_path, normalized_path)
            print(f"Renamed: {file_name} -> {normalized_name}")



if __name__ == "__main__":

    # Example usage
    directory = "../static/docs"  # Path to your directory
    normalize_files_in_directory(directory)

    # Directory containing the PDFs to be checked
    # input_directory = "../static/docs"
    # output_directory = "../static/docs/SPLIT"
    #
    # # Create the output directory if it doesn't exist
    # os.makedirs(output_directory, exist_ok=True)
    #
    # # Initialize part counter
    # part_counter = 1
    #
    # # Get and sort files in natural order
    # file_list = natsorted(os.listdir(input_directory))
    #
    # # Iterate through all sorted PDF files in the input directory
    # for file_name in file_list:
    #     if file_name.endswith(".pdf"):
    #         input_path = os.path.join(input_directory, file_name)
    #         part_counter = split_pdf_into_seven(input_path, output_directory, part_counter)
