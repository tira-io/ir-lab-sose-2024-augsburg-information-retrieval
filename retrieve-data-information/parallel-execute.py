import subprocess
from concurrent.futures import ThreadPoolExecutor
import os

def get_splitted_docfiles(directory_path):
    try:
        files = os.listdir(directory_path)
        return files
    except Exception as e:
        print(f"Error reading directory {directory_path}: {e}")
        return []


def run_script(input_param):
    result = subprocess.run(['python', 'retrieve-paper-information.py', str(input_param)], capture_output=True, text=True)
    return result.stdout



if __name__ == '__main__':
    directory_path = "data/splitted_docs/"
    files = get_splitted_docfiles(directory_path)



    with ThreadPoolExecutor() as executor:
        results = executor.map(run_script, files)

    for result in results:
        print(result)



