import os
import subprocess


def convert_notebook_to_markdown(notebook_path):
    """
    Converts a Jupyter Notebook to Markdown format.
    """
    if not notebook_path.endswith(".ipynb"):
        print("Please provide a valid .ipynb file.")
        return

    try:
        # Run the nbconvert command
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "markdown", notebook_path],
            check=True
        )
        print(f"Successfully converted {notebook_path} to Markdown.")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def main():
    # Directory containing your notebooks
    notebook_dir = "./notebooks"  # Adjust to your directory path

    # Iterate through all notebooks in the directory
    for file_name in os.listdir(notebook_dir):
        if file_name.endswith(".ipynb"):
            notebook_path = os.path.join(notebook_dir, file_name)
            convert_notebook_to_markdown(notebook_path)


if __name__ == "__main__":
    main()
