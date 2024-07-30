from datasets import load_dataset
import json

# Load MS-MARCO dataset
dataset = load_dataset('ms_marco', 'v1.1', split='train')

# Function to get the first relevant passage
def get_first_relevant_passage(dataset_entry):
    passages = dataset_entry['passages']['passage_text']
    is_selected = dataset_entry['passages']['is_selected']
    
    # Find the first relevant passage
    for passage, selected in zip(passages, is_selected):
        if selected:
            return passage
    
    # Return None if no passages are selected
    return None

def process_entries(dataset):
    saved_queries_and_passages = []
    
    for entry in dataset:
        first_relevant_passage = get_first_relevant_passage(entry)
        
        # Check if a relevant passage was found
        if first_relevant_passage is not None:
            saved_queries_and_passages.append({
                "query": entry['query'],
                "passage": first_relevant_passage
            })
    
    # Save the results to a JSON file
    with open('query-expansion/ms-marco_queries_and_passages.json', 'w') as f:
        json.dump(saved_queries_and_passages, f, indent=4)
    
    print("Results have been saved to 'queries_and_passages.json'.")

process_entries(dataset)  # Process the entire dataset

