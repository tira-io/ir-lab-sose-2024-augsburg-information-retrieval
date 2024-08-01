import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import pyterrier as pt


class Doc2Query: 

    def __init__(self, model_id, temperatur, promting_technique):
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
        self.promting_technique = promting_technique


    def expandDocumentsByQueries(self, documents_df):
        '''Expands the documents by queries. Return the extended pyterrier dataset.'''
        # Next line is just for testing
        documents_df = documents_df.head(50)
        queries = documents_df['text'].apply(self.createQueries)
        expaneded_documents_df = documents_df.copy()
        expaneded_documents_df['text'] = documents_df.apply(lambda row: f"{row['text']} {queries[row.name]}", axis=1)
        return expaneded_documents_df
        
        
    
    def getDocumentsDfFromPtDataset(self, pt_dataset): 
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
        prompt = self.getPrompt(input_text)

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


    def getPromt(self, input_text): 
        if self.promting_technique == "Zero-Shot": 
            prompt = self.getZeroShotPrompt(input_text)
        elif self.promting_technique == "One-Shot": 
            prompt = self.getOneShotPrompt(input_text)
        elif self.promting_technique == "Few-Shot": 
            prompt = self.getFewShotPrompt(input_text)
        else: 
            print("Method not specified for this promting technique. Best regards, Georg")
            prompt = None
        return prompt

    def getZeroShotPrompt(self, input_text): 
        prompt = f"Generate three queries based on the following text:\n\n{input_text}\n\nQuery 1:\nQuery 2:\nQuery 3:"
        return prompt
    
    def getOneShotPrompt(self, input_text): 
        oneShotExample = "Inverted indexes for phrases and strings\n\n\n ABSTRACTInverted indexes are the most fundamental and widely used data structures in information \
                            retrieval. For each unique word occurring in a document collection, the inverted index stores a list of the documents in which this word occurs. \
                            Compression techniques are often applied to further reduce the space requirement of these lists. However, the index has a shortcoming, in that only \
                            predefined pattern queries can be supported efficiently. In terms of string documents where word boundaries are undefined, if we have to index all \
                            the substrings of a given document, then the storage quickly becomes quadratic in the data size. Also, if we want to apply the same type of indexes \
                            for querying phrases or sequence of words, then the inverted index will end up storing redundant information. In this paper, we show the first set \
                            of inverted indexes which work naturally for strings as well as phrase searching. The central idea is to exclude document d in the inverted list of \
                            a string P if every occurrence of P in d is subsumed by another string of which P is a prefix. With this we show that our space utilization is close \
                            to the optimal. Techniques from succinct data structures are deployed to achieve compression while allowing fast access in terms of frequency and \
                            document id based retrieval. Compression and speed tradeoffs are evaluated for different variants of the proposed index. For phrase searching, \
                            we show that our indexes compare favorably against a typical inverted index deploying position-wise intersections. We also show efficient top-k \
                            based retrieval under relevance metrics like frequency and tf-idf.\n\nQuery 1: reverse indexing	\nQuery 2: index compression techniques	 \nQuery 3: \
                            principle of a information retrieval indexing"
        prompt = f"The task is to generate queries based on a given text. Look at this example:\n\n{oneShotExample}\n\nGenerate three queries just like in the example above \
                    based on the following text:\n\n{input_text}\n\nQuery 1:\nQuery 2:\nQuery 3:"
        return prompt
    
    def getFewShotPrompt(self, input_text): 
        fewShotExample = "Inverted indexes for phrases and strings\n\n\n ABSTRACTInverted indexes are the most fundamental and widely used data structures in information \
                            retrieval. For each unique word occurring in a document collection, the inverted index stores a list of the documents in which this word occurs. \
                            Compression techniques are often applied to further reduce the space requirement of these lists. However, the index has a shortcoming, in that only \
                            predefined pattern queries can be supported efficiently. In terms of string documents where word boundaries are undefined, if we have to index all \
                            the substrings of a given document, then the storage quickly becomes quadratic in the data size. Also, if we want to apply the same type of indexes \
                            for querying phrases or sequence of words, then the inverted index will end up storing redundant information. In this paper, we show the first set \
                            of inverted indexes which work naturally for strings as well as phrase searching. The central idea is to exclude document d in the inverted list of \
                            a string P if every occurrence of P in d is subsumed by another string of which P is a prefix. With this we show that our space utilization is close \
                            to the optimal. Techniques from succinct data structures are deployed to achieve compression while allowing fast access in terms of frequency and \
                            document id based retrieval. Compression and speed tradeoffs are evaluated for different variants of the proposed index. For phrase searching, \
                            we show that our indexes compare favorably against a typical inverted index deploying position-wise intersections. We also show efficient top-k \
                            based retrieval under relevance metrics like frequency and tf-idf.\n\nQuery 1: reverse indexing	\nQuery 2: index compression techniques	 \nQuery 3: \
                            principle of a information retrieval indexing\n\n\n \
                            When do Recommender Systems Work the Best?: The Moderating Effects of Product Attributes and Consumer Reviews on Recommender Performance\n\n\n ABSTRACTWe \
                            investigate the moderating effect of product attributes and consumer reviews on the efficacy of a collaborative filtering recommender system on an e-commerce \
                            site. We run a randomized field experiment on a top North American retailer's website with 184,375 users split into a recommendertreated group and a control \
                            group with 37,215 unique products in the dataset. By augmenting the dataset with Amazon Mechanical Turk tagged product attributes and consumer review data \
                            from the website, we study their moderating influence on recommenders in generating conversion.We first confirm that the use of recommenders increases the \
                            baseline conversion rate by 5.9%. We find that the recommenders act as substitutes for high average review ratings with the effect of using recommenders \
                            increasing the conversion rate as much as about 1.4 additional average star ratings. Additionally, we find that the positive impacts on conversion from \
                            recommenders are greater for hedonic products compared to utilitarian products while searchexperience quality did not have any impact. We also find that \
                            the higher the price, the lower the positive impact of recommenders, while having lengthier product descriptions and higher review volumes increased the \
                            recommender's effectiveness. More findings are discussed in the Results.For managers, we 1) identify the products and product attributes for which the \
                            recommenders work well, 2) show how other product information sources on e-commerce sites interact with recommenders. Additionally, the insights from the \
                            results could inform novel recommender algorithm designs that are aware of strength and shortcomings. From an academic standpoint, we provide insight into \
                            the underlying mechanism behind how recommenders cause consumers to purchase.\n\nQuery 1: recommenders influence on users \nQuery 2: consumer product reviews \
                            \nQuery 3: recommendation systems"
        prompt = f"The task is to generate queries based on a given text. Look at these examples:\n\n{fewShotExample}\n\nGenerate three queries just like in the examples above \
                    based on the following text:\n\n{input_text}\n\nQuery 1:\nQuery 2:\nQuery 3:"
        return prompt

    def test(self):
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

        inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
        outputs = model.generate(**inputs)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ['Pour a cup of bolognese into a large bowl and add the pasta']