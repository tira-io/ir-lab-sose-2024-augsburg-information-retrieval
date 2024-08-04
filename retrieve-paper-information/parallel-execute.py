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


def parallel_execute(input_param):
    try:
        result = subprocess.run(['python3', 'retrieve-paper-information/retrieve-paper-information.py', str(input_param)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        output = result.stdout.decode('utf-8')
        print(output)       
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"


if __name__ == '__main__':
    print("started")
    directory_path = "data/splitted_docs/"
    files = get_splitted_docfiles(directory_path)
    files.sort()

    parallel_api_calls = 1
    for i in range(0, len(files), parallel_api_calls):
    
        with ThreadPoolExecutor() as executor:
            executor.map(parallel_execute, files[i:i+parallel_api_calls])
        
        print(f'Successfully retrieved the documents: {i} to {i+parallel_api_calls-1}')
    
    print("completed finally")



