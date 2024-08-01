import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import pyterrier as pt


class Doc2Query: 

    def __init__(self, model_id, temperatur):
        '''
        Initialize the Doc2Query class and load the tokenizer and the LLM. 
        Model ids might be "mistralai/Mixtral-8x7B-v0.1" or "meta-llama/Llama-2-7b-chat-hf" or "gpt2" or "google/flan-t5-small"
        
        '''
        self.model_id = model_id
        if model_id == "google/flan-t5-small":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)  # , load_in_4bit=True 
        self.temperatur = temperatur


    def expandDocumentsByQueries(self, pt_dataset_name):
        '''Expands the documents by queries. Return the extended pyterrier dataset.'''
        pt_dataset = pt.get_dataset(pt_dataset_name)
        text_df = self.getTextDfFromPtDataset(pt_dataset)
        # Just for testing
        #text_df = text_df.head()
        queries = text_df['text'].apply(self.createQueries)
        text_df['text'] = text_df.apply(lambda row: f"{row['text']} {queries[row.name]}", axis=1)
        return text_df
        
        
    
    def getTextDfFromPtDataset(self, pt_dataset): 
        '''Given a pyterrier dataset, the texts and document numbers get extracted into a dataframe, which gets returned'''
         # Get the documents generator
        documents = pt_dataset.get_corpus_iter()
        # Extract docno and text into a DataFrame
        doc_list = []
        for doc in documents:
            doc_list.append(doc)
        df = pd.DataFrame(doc_list)
        return df
    

    def createQueries(self, input_text): 
        '''Promts the LLM to get queries for a given document'''
        # Define the input text and the prompt for query generation
        prompt = f"Generate three queries based on the following text:\n\n{input_text}\n\nQuery 1:\nQuery 2:\nQuery 3:"

        if self.model_id == "mistralai/Mixtral-8x7B-v0.1": 
            # Encode the prompt and generate text
            inputs = self.tokenizer(prompt, return_tensors="pt").to(0)
            outputs = self.model.generate(**inputs, max_new_tokens=20)
            # Decode the generated text
            queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif self.model_id == "meta-llama/Llama-2-7b-chat-hf": 
            # Encode the prompt and generate text
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs["input_ids"], max_length=100)
            # Decode the generated text
            queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif self.model_id == "gpt2": 
            # Encode the prompt and generate text
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True, temperature=self.temperatur)
            # Decode the generated text
            queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        elif self.model_id == "google/flan-t5-small": 
            # Encode the prompt and generate text
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            # Decode the generated text
            queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            queries = None
            print("Method not specified for this model. Please check model id. Best regards, Georg")

        return queries





    def test(self):
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

        inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
        outputs = model.generate(**inputs)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ['Pour a cup of bolognese into a large bowl and add the pasta']