import torch
import torch.nn as nn

import numpy as np

class VAE(nn.Module):
    def __init__(self, encoder, decoder, device='cpu'):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)     
        z = mean + var * epsilon
        return z
        
                
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
    
    def inference_theta(self, x):
        with torch.no_grad():
            mean, log_var = self.encoder(x)
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
            theta = torch.softmax(z,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()


class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h = self.LeakyReLU(self.lin1(x))
        h = self.LeakyReLU(self.lin2(h))
        mean = self.mean(h)
        log_var = self.var(h)
        
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        
    def forward(self, x):
        h = self.LeakyReLU(self.lin1(x))
        h = self.LeakyReLU(self.lin2(h))
        
        x_hat = torch.softmax(self.out(h), dim=1)
        return x_hat


class SparseDataset():

    def __init__(self, mat_csr, device='cpu'):
        self.dim = mat_csr.shape
        self.device = torch.device(device)

        self.indptr = torch.tensor(mat_csr.indptr, dtype=torch.int64, device=self.device)
        self.indices = torch.tensor(mat_csr.indices, dtype=torch.int64, device=self.device)
        self.data = torch.tensor(mat_csr.data, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.dim[0]

    def __getitem__(self, idx):
        obs = torch.zeros((self.dim[1],), dtype=torch.float32, device=self.device)
        ind1,ind2 = self.indptr[idx],self.indptr[idx+1]
        obs[self.indices[ind1:ind2]] = self.data[ind1:ind2]
        return obs
    
def topic_cos_sim(keyFreq, posting, entryStats, collStats, vae):
    print("hi")
    print(posting.getId())
    print(keyFreq)
    print("hi2")
    query_bow = []
    doc_bow = []
    vae.eval()

    with torch.no_grad():
        q_theta = vae.inference_theta(query_bow)
        doc_theta = vae.inference_theta(doc_bow)
    
    return np.dot(q_theta, doc_theta) / (np.norm(q_theta) * np.norm(doc_theta))