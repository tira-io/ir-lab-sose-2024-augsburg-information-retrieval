import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import pyterrier as pt


class Doc2Query: 

    def __init__(self, model_id):
        '''
        Initialize the Doc2Query class and load the tokenizer and the LLM. 
        Model ids might be "mistralai/Mixtral-8x7B-v0.1" or "meta-llama/Llama-2-7b-chat-hf"
        
        '''
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)  # Geht das mit Llama auch in 4bit?


    def expandDocumentsByQueries(self, pt_dataset_name):
        '''Expands the documents by queries. Return the extended pyterrier dataset.'''
        # Initialize PyTerrier
        #if not pt.started():
        #    pt.init()
        #pt_dataset_name = 'irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training'
        pt_dataset = pt.get_dataset(pt_dataset_name)
        text_df = self.getTextDfFromPtDataset(pt_dataset)
        queries = text_df['text'].apply(self.createQueries)
        #text_df['text'] = text_df['text'] + ' ' + queries
        text_df['text'] = text_df.apply(lambda row: f"{row['text']} {queries[row.name]}", axis=1)
        return text_df
        
        #TODO: Return pyterrier dataset

        # Create a new modifierd pyterrier dataset
        #modified_df = text_df
        # Define a custom dataset class
        #class CustomModifiedDataset(pt.Dataset):
           # def __init__(self, documents_df):
                #self.documents_df = documents_df

            #def get_corpus_iter(self):
                # Yield each modified document as a dictionary
                #for index, row in self.documents_df.iterrows():
                    #yield {'docno': row['docno'], 'text': row['text']}

        # Return the custom dataset instance
        #return CustomModifiedDataset(modified_df)
        
        

    def getTextDfFromPtDataset(self, pt_dataset): 
        '''Given a pyterrier dataset, the texts and document numbers get extracted into a dataframe, which gets returned'''
         # Get the documents generator
        documents = pt_dataset.get_corpus_iter()
        # Extract docno and text into a DataFrame
        doc_list = []
        for doc in documents:
            docno = doc['docno']
            text = doc['text']
            doc_list.append({'docno': docno, 'text': text})
        df = pd.DataFrame(doc_list)
        return df
    

    def createQueries(self, text): 
        #TODO: Not just one querie but multiple (three)
        '''Promts the LLM to get queries for a given document'''
        prompt = "Generate queries about the following text: " + text
        if self.model_id == "mistralai/Mixtral-8x7B-v0.1": 
            # Encode the prompt and generate text
            inputs = self.tokenizer(prompt, return_tensors="pt").to(0)
            outputs = self.model.generate(**inputs, max_new_tokens=20)
            # Decode the generated text
            queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif self.model_name == "meta-llama/Llama-2-7b-chat-hf": 
            # Encode the prompt and generate text
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs["input_ids"], max_length=100)
            # Decode the generated text
            queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            queries = None
            print("Method not specified for this model. Please check model id.")

        return queries






    def test(self):
        model_id = "mistralai/Mixtral-8x7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

        text = "Hello my name is"
        inputs = tokenizer(text, return_tensors="pt").to(0)

        outputs = model.generate(**inputs, max_new_tokens=20)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))