# A prepared image with python3.10, java 11, ir_datasets, tira, and PyTerrier installed 
FROM webis/ir-lab-wise-2023:0.0.4

# Update the tira command to use the latest version
RUN pip3 uninstall -y tira \
	&& pip3 install tira

RUN pip3 install torch \
	&& pip3 install transformers \
	&& pip3 install colbert \
	&& pip3 install torch

ADD . /app

