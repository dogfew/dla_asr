import os
import wget
import gzip
import shutil


def process_file(url, output_path, is_gzip=False):
    if not os.path.exists(output_path):
        print(f"Downloading file from {url} to {output_path}.")
        wget.download(url, output_path)

        if is_gzip:
            uncompressed_path = output_path.replace(".gz", "")
            if not os.path.exists(uncompressed_path):
                with gzip.open(output_path, "rb") as f_zipped:
                    with open(uncompressed_path, "wb") as f_unzipped:
                        shutil.copyfileobj(f_zipped, f_unzipped)
                print(f"Unzipped file to {uncompressed_path}.")
            output_path = uncompressed_path  # Update output_path to uncompressed file path for lowercase conversion

        with open(output_path, "r") as f_original:
            content = f_original.read().lower()
        with open(output_path, "w") as f_lower:
            f_lower.write("\n".join([i for i in content.splitlines() if "'" not in i]))
        print(f"Converted file to lowercase at {output_path}.")
    else:
        print(f"{output_path} already exists.")


def get_language_model() -> tuple[str, str]:
    """
    :returns: (Language Model Path, Vocab Path)
    """
    # Ensure the output directory exists
    os.makedirs("language_models", exist_ok=True)

    # Process language model
    lm_url = "http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
    lm_output_path = os.path.join("language_models", "3-gram.pruned.1e-7.arpa.gz")
    process_file(lm_url, lm_output_path, is_gzip=True)

    # Process vocab
    vocab_url = "https://www.openslr.org/resources/11/librispeech-vocab.txt"
    vocab_output_path = os.path.join("language_models", "librispeech-vocab.txt")
    process_file(vocab_url, vocab_output_path)

    return lm_output_path, vocab_output_path
