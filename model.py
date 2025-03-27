import torch
import torch.nn as nn
import torch.nn.functional as F

# Three different activation function uses in the ZINB-based denoising autoencoder.
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e4)
DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e3)
PiAct = lambda x: 1/(1+torch.exp(-1 * x))

# A general GCN layer.
class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = torch.tanh(output)
        return output
    
# A dot product operation uses in the decoder of GAE. 
def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

# A random Gaussian noise uses in the ZINB-based denoising autoencoder.
class GaussianNoise(nn.Module):
    def __init__(self, device, sigma=1, is_relative_detach=True):
        super(GaussianNoise,self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, device = device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

class DynamicMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=8, dropout=0.2, use_dynamic_weight=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.use_dynamic_weight = use_dynamic_weight

        self.to_queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_values = nn.Linear(emb_dim, emb_dim, bias=False)

        # Dynamic combination weights (learnable per-head weights)
        if use_dynamic_weight:
            self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

        # Final output projection
        self.unify_heads = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, emb_dim)
            y: Optional second input (if None, becomes self-attention)
            mask: Optional attention mask
        Returns:
            Output tensor (batch_size, seq_len, emb_dim)
        """
        if y is None:
            y = x  # Self-attention

        batch_size, seq_len, emb_dim = x.shape

        # Project Q, K, V (split into heads)
        queries = self.to_queries(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.to_keys(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.to_values(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, values)

        # Dynamic head combination (learned weights)
        if self.use_dynamic_weight:
            out = out * self.head_weights.view(1, self.num_heads, 1, 1)

        # Concatenate heads and project back to emb_dim
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)
        out = self.unify_heads(out)

        return out

# Final model
class scDAEC(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, heads, device):
        super(scDAEC, self).__init__()

        # autoencoder for intra information
        #self.dropout = nn.Dropout(0.2)
        self.Gnoise = GaussianNoise(sigma=0.1, device=device)
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.BN_1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN_2 = nn.BatchNorm1d(n_enc_2)
        self.z_layer = nn.Linear(n_enc_2, n_z)
        
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
   
        self.calcu_pi = nn.Linear(n_dec_2, n_input)
        self.calcu_disp = nn.Linear(n_dec_2, n_input)
        self.calcu_mean = nn.Linear(n_dec_2, n_input)
       
        self.gnn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_2 = GNNLayer(n_enc_2, n_z)

        self.attn1 = DynamicMultiHeadAttention(n_enc_2,heads=heads)
        self.attn2 = DynamicMultiHeadAttention(n_z,heads=heads)


    def forward(self, x, adj):  
        enc_h1 = self.BN_1(F.relu(self.enc_1(self.Gnoise(x))))
        # enc_h1 = (self.attn1(enc_h1, h1)).squeeze(0) + enc_h1
        h1 = self.gnn_1(enc_h1, adj)
        h2 = self.gnn_2(h1, adj)
        enc_h2 = self.BN_2(F.relu(self.enc_2(self.Gnoise(enc_h1))))
        enc_h2 = (self.attn1(enc_h2, h1)).squeeze(0)+enc_h2
        z = self.z_layer(self.Gnoise(enc_h2))
        z = (self.attn2(z, h2)).squeeze(0)+z
        #decoder
        A_pred = dot_product_decode(h2)
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        
        pi = PiAct(self.calcu_pi(dec_h2))
        mean = MeanAct(self.calcu_mean(dec_h2))
        disp = DispAct(self.calcu_disp(dec_h2))        
        return z, A_pred, pi, mean, disp