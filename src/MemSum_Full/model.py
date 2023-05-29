import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions import Categorical


class AddMask(nn.Module):
    def __init__(self, pad_index):
        super().__init__()
        self.pad_index = pad_index

    def forward(self, x):
        mask = x == self.pad_index
        return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        dim_per_head = int(embed_dim/num_heads)
        
        self.ln_q = nn.Linear(embed_dim, num_heads * dim_per_head)
        self.ln_k = nn.Linear(embed_dim, num_heads * dim_per_head)
        self.ln_v = nn.Linear(embed_dim, num_heads * dim_per_head)

        self.ln_out = nn.Linear(num_heads * dim_per_head, embed_dim)

        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
    
    def forward(self, q, k, v, mask=None):
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(q.size(0), q.size(1),  self.num_heads, self.dim_per_head).transpose(1, 2)
        k = k.view(k.size(0), k.size(1),  self.num_heads, self.dim_per_head).transpose(1, 2)
        v = v.view(v.size(0), v.size(1),  self.num_heads, self.dim_per_head).transpose(1, 2)

        a = self.scaled_dot_product_attention(q, k, mask)
        new_v = a.matmul(v)
        new_v = new_v.transpose(1, 2).contiguous()
        new_v = new_v.view(new_v.size(0), new_v.size(1), -1)
        new_v = self.ln_out(new_v)
        return new_v

    def scaled_dot_product_attention(self, q, k, mask=None):
        a = q.matmul(k.transpose(2, 3)) / math.sqrt(q.size(-1))

        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e9)
        a = F.softmax(a, dim=-1)
        return a


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ln1 = nn.Linear(embed_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        net = F.relu(self.ln1(x))
        out = self.ln2(net)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask, dropout_rate=0.):
        short_cut = x
        net = F.dropout(self.mha(x, x, x, mask), p=dropout_rate)
        net = self.norm1(short_cut + net)
        short_cut = net
        net = F.dropout(self.feed_forward(net), p=dropout_rate)
        net = self.norm2(short_cut + net)
        return net


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.masked_mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, encoder_output, x, src_mask, trg_mask, dropout_rate=0.):
        short_cut = x
        net = F.dropout(self.masked_mha(x, x, x, trg_mask), p=dropout_rate)  # 注意剩余句子
        net = self.norm1(short_cut + net)
        short_cut = net
        net = F.dropout(self.mha(net, encoder_output, encoder_output, src_mask), p=dropout_rate)  # 注意历史
        net = self.norm2(short_cut + net)
        short_cut = net
        net = F.dropout(self.feed_forward(net), p=dropout_rate)
        net = self.norm3(short_cut + net)
        return net 


class MultiHeadPoolingLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = int(embed_dim/num_heads)
        self.ln_attention_score = nn.Linear(embed_dim, num_heads)
        self.ln_value = nn.Linear(embed_dim,  num_heads * self.dim_per_head)
        self.ln_out = nn.Linear(num_heads * self.dim_per_head, embed_dim)

    def forward(self, input_embedding, mask=None):
        a = self.ln_attention_score(input_embedding)
        v = self.ln_value(input_embedding)
        
        a = a.view(a.size(0), a.size(1), self.num_heads, 1).transpose(1, 2)
        v = v.view(v.size(0), v.size(1),  self.num_heads, self.dim_per_head).transpose(1, 2)
        a = a.transpose(2, 3)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e9)
        a = F.softmax(a, dim=-1)

        new_v = a.matmul(v)
        new_v = new_v.transpose(1, 2).contiguous()
        new_v = new_v.view(new_v.size(0), new_v.size(1), -1).squeeze(1)
        new_v = self.ln_out(new_v)
        return new_v


class MultiHeadUpSamPoolingLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):  # 200 8
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = embed_dim
        self.ln_attention_score = nn.Linear(embed_dim, num_heads)
        self.ln_value = nn.Linear(embed_dim, num_heads * embed_dim)
        self.ln_out = nn.Linear(num_heads * embed_dim, num_heads * embed_dim)

    def forward(self, input_embedding, mask=None):
        a = self.ln_attention_score(input_embedding)
        v = self.ln_value(input_embedding)

        a = a.view(a.size(0), a.size(1), self.num_heads, 1).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.dim_per_head).transpose(1, 2)
        a = a.transpose(2, 3)
        if mask is not None:
            a = a.masked_fill(mask.unsqueeze(1).unsqueeze(1), -1e9)
        a = F.softmax(a, dim=-1)

        new_v = a.matmul(v)
        new_v = new_v.transpose(1, 2).contiguous()
        new_v = new_v.view(new_v.size(0), new_v.size(1), -1).squeeze(1)
        new_v = self.ln_out(new_v)
        return new_v


class LocalSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, pad_index, embed_dim, num_heads, hidden_dim, num_enc_layers, pretrained_word_embedding):
        super().__init__()
        self.addmask = AddMask(pad_index)
      
        self.rnn = nn.LSTM(embed_dim, embed_dim, 2, batch_first=True, bidirectional=True)
        self.mh_pool = MultiHeadPoolingLayer(2*embed_dim, num_heads)
        self.norm_out = nn.LayerNorm(2*embed_dim)
        self.ln_out = nn.Linear(2*embed_dim, embed_dim)

        if pretrained_word_embedding is not None:
            ## make sure the pad embedding is 0
            pretrained_word_embedding[pad_index] = 0
            self.register_buffer("word_embedding", torch.from_numpy(pretrained_word_embedding))
        else:
            self.register_buffer("word_embedding", torch.randn( vocab_size, embed_dim))

    """
    input_seq 's shape:  batch_size x seq_len 
    """
    def forward(self, input_seq, dropout_rate=0.):
        self.rnn.flatten_parameters()
        mask = self.addmask(input_seq)
        ## batch_size x seq_len x embed_dim
        net = self.word_embedding[input_seq]
        net, _ = self.rnn(net)
        net = self.ln_out(F.relu(self.norm_out(self.mh_pool(net, mask))))
        return net


class GlobalContextEncoder(nn.Module):
    def __init__(self, embed_dim,  num_heads, hidden_dim, num_dec_layers):
        super().__init__()
        self.rnn = nn.LSTM(embed_dim, embed_dim, 2, batch_first=True, bidirectional=True)
        self.norm_out = nn.LayerNorm(2*embed_dim)
        self.ln_out = nn.Linear(2*embed_dim, embed_dim)

    def forward(self, sen_embed, doc_mask, dropout_rate=0.):
        self.rnn.flatten_parameters()
        net, _ = self.rnn(sen_embed)
        net = self.ln_out(F.relu(self.norm_out(net)))
        return net


class ExtractionContextDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_dec_layers):
        super().__init__()
        self.layer_list = nn.ModuleList([TransformerDecoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_dec_layers)])

    def forward(self, sen_embed, remaining_mask, extraction_mask, dropout_rate=0.):
        net = sen_embed
        for layer in self.layer_list:
            net = layer(sen_embed, net, remaining_mask, extraction_mask, dropout_rate)
        return net


class Extractor(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm_input = nn.LayerNorm(4*embed_dim)
        
        self.ln_hidden1 = nn.Linear(4*embed_dim, 3*embed_dim)
        self.norm_hidden1 = nn.LayerNorm(3*embed_dim)
        
        self.ln_hidden2 = nn.Linear(3*embed_dim, 2*embed_dim)
        self.norm_hidden2 = nn.LayerNorm(2*embed_dim)

        self.ln_hidden3 = nn.Linear(2 * embed_dim, embed_dim)
        self.norm_hidden3 = nn.LayerNorm(embed_dim)

        self.ln_out = nn.Linear(embed_dim, 1)

        self.mh_pool = MultiHeadPoolingLayer(embed_dim, num_heads)
        self.norm_pool = nn.LayerNorm(embed_dim)
        self.ln_stop = nn.Linear(embed_dim, 1)

    def forward(self, sen_embed, relevance_embed, topic_embed, redundancy_embed, extraction_mask, dropout_rate=0.):
        if redundancy_embed is None:
            redundancy_embed = torch.zeros_like(sen_embed)
        net = self.norm_input(F.dropout(torch.cat([sen_embed, relevance_embed, topic_embed, redundancy_embed], dim=2), p=dropout_rate))
        net = F.relu(self.norm_hidden1(F.dropout(self.ln_hidden1(net), p=dropout_rate)))
        net = F.relu(self.norm_hidden2(F.dropout(self.ln_hidden2(net), p=dropout_rate)))
        hidden_net = F.relu(self.norm_hidden3(F.dropout(self.ln_hidden3(net), p=dropout_rate)))
        
        p = self.ln_out(hidden_net).sigmoid().squeeze(2)

        net = F.relu(self.norm_pool(F.dropout(self.mh_pool(hidden_net, extraction_mask), p=dropout_rate)))
        p_stop = self.ln_stop(net).sigmoid().squeeze(1)

        return p, p_stop
        # return p


class VAE(nn.Module):
    def __init__(self, encode_dims, dropout):
        super().__init__()

        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i + 1])
            for i in range(len(encode_dims) - 2)
        })

        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])

        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)

        self.in_features = encode_dims[0]  # 输入句子的数量
        self.n_topic = encode_dims[-1]  # 主题数量

        self.fc1 = nn.Linear(200, 1)

        self.rho = nn.Linear(200, self.in_features)
        self.alpha = nn.Linear(200, self.n_topic)

        self.norm_input = nn.LayerNorm(400)
        self.ln_hidden = nn.Linear(400, 200)
        self.norm_hidden = nn.LayerNorm(200)

    def encode(self, x):
        hid = x
        for i, layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, redundancy_embed):
        wght_dec = self.alpha(self.rho.weight)  # self.rho.weight 句子总数 * 向量维度     wght_dec  句子总数 * 主题数量
        beta = F.softmax(wght_dec, dim=0).transpose(1, 0)
        # 主题还原成文档
        res = torch.mm(z, beta)
        logits = torch.log(res+1e-6)

        topic_vec = self.alpha.weight  # 50 * 200 主题数量和向量维度
        h_z = np.argmax(z.detach().cpu().numpy(), axis=1)
        h_z = topic_vec[h_z]

        if redundancy_embed is None:
            redundancy_embed = torch.zeros_like(h_z)

        net = self.norm_input(self.dropout(torch.cat([h_z, redundancy_embed], dim=1)))
        net = F.relu(self.norm_hidden(self.dropout(self.ln_hidden(net))))

        return logits, net

    def loss(self, bows, bows_recon, mus, log_vars):
        logsoftmax = torch.log_softmax(bows_recon, dim=1)
        rec_loss = -1.0 * torch.sum(bows * logsoftmax, dim=1)

        kl_div = -0.5 * torch.sum((1 + log_vars - mus.pow(2) - log_vars.exp()), dim=1)

        loss = rec_loss + kl_div

        return loss

    def forward(self, x, mask, dropout_rate, redundancy_embed=None):  # 输入X维度 ： batch * sent_num * 200

       net = self.fc1(x).squeeze(-1).masked_fill(mask, -1e9).softmax(dim=1)

       mu, log_var = self.encode(net)
       _theta = self.reparameterize(mu, log_var)
       theta = _theta.softmax(dim=1)
       x_reconst, h_z = self.decode(theta, redundancy_embed)

       loss = self.loss(net, x_reconst, mu, log_var)

       return h_z, loss

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(F.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.ft = nn.Linear(in_dim, out_dim, bias=False)
        self.fg = nn.Linear(in_dim, out_dim, bias=False)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.attn_fc_1 = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, topic_embed, global_context_embed, mask):
        src = self.ft(topic_embed)
        dst = self.fg(global_context_embed)
        z = F.leaky_relu(self.attn_fc(torch.cat([src, dst], dim=-1)).squeeze(-1).masked_fill(mask, -1e9))
        a = F.softmax(z, dim=-1)
        # 更新全局表示
        u = torch.tanh(a.unsqueeze(-1) * self.fc(global_context_embed))

        # 更新主题
        z = F.leaky_relu(self.attn_fc_1(torch.cat([src, dst], dim=-1)).squeeze(-1).masked_fill(mask, -1e9))
        a = F.softmax(z, dim=-1)

        topic = torch.tanh(torch.sum(a.unsqueeze(-1) * self.fc(global_context_embed), dim=1))
        return u, topic


class MultiHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, layer):
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(layer(in_dim, out_dim))  # [n_nodes, hidden_size]
        self.dropout = nn.Dropout(attn_drop_out)

    def forward(self, topic_embed, global_context_embed, mask):
        gl_out = []
        topic_out = []
        for attn_head in self.heads:
            head_out, head_out_ = attn_head(self.dropout(topic_embed), self.dropout(global_context_embed), mask)
            gl_out.append(head_out)
            topic_out.append(head_out_)

        gl = torch.cat(gl_out, dim=-1)
        topic = torch.cat(topic_out, dim=-1)

        return gl, topic


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out):
        super(GAT, self).__init__()
        self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, layer=GATLayer)
        self.norm = nn.LayerNorm(in_dim)
        self.pw = PositionwiseFeedForward(in_dim, 3 * in_dim)

    def forward(self, topic_embed, global_context_embed, mask):  # 输入X维度 ： batch * sent_num * 200

        batch = topic_embed.size(0)
        hidden = topic_embed.size(1)
        num_sent = global_context_embed.size(1)

        topic_embed = topic_embed.repeat(1, num_sent).view(batch, num_sent, hidden)

        global_context_embed, topic_embed = self.layer(topic_embed, global_context_embed, mask)

        topic_embed = topic_embed.repeat(1, num_sent).view(batch, num_sent, hidden)

        global_context_embed, topic_embed = self.layer(self.norm(topic_embed), self.norm(global_context_embed), mask)

        topic_embed = topic_embed.repeat(1, num_sent).view(batch, num_sent, hidden)

        return self.pw(global_context_embed), self.pw(topic_embed)
        # return self.norm(global_context_embed), self.norm(topic_embed)

