from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

MODEL_PATH = "/hpc/gpfs2/scratch/g/coling/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, device_map="auto")

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
#model.to(device)i

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
text = tokenizer.batch_decode(generated_ids)[0]
print(prompt, text)
