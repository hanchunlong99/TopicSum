import torch

from datautils1 import *
from model1 import *
from utils import *
import pickle
from tqdm import tqdm
import time
import os, sys
from torch.distributions import Categorical
from rouge_score import rouge_scorer
import copy
import argparse

def update_moving_average(m_ema, m, decay):
    with torch.no_grad():
        param_dict_m_ema = m_ema.module.parameters() if isinstance(m_ema, nn.DataParallel) else m_ema.parameters()
        param_dict_m = m.module.parameters() if isinstance(m, nn.DataParallel) else m.parameters()
        for param_m_ema, param_m in zip(param_dict_m_ema, param_dict_m):
            param_m_ema.copy_(decay * param_m_ema + (1-decay) * param_m)

def LOG(info, end="\n"):
    global log_out_file
    with open(log_out_file, "a") as f:
        f.write(info + end)

def load_corpus(fname, is_training):
    corpus = []
    n = []
    with open(fname, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if len(data["text"]) < 7 or len(data["summary"]) == 0:
                continue
            if is_training:
                if len(data["indices"]) == 0 or len(data["score"]) == 0:
                    continue
                assert len(data["indices"]) == len(data["score"])
                for i in data["indices"]:
                    for j in i:
                        if j == len(data["text"]):
                            print("1")
                        if j > len(data["text"]):
                            print("1")

            corpus.append(data)
            n.append(len(data['text']))
    return corpus

parser = argparse.ArgumentParser()

parser.add_argument("-training_corpus_file_name" )
parser.add_argument("-validation_corpus_file_name" )
parser.add_argument("-model_folder")
parser.add_argument("-log_folder")
parser.add_argument("-vocabulary_file_name" )
parser.add_argument("-pretrained_unigram_embeddings_file_name")

# graph
parser.add_argument("-graph_num_heads", type=int, default=8)
parser.add_argument("-graph_layer", type=int, default=3)
parser.add_argument("-graph_hidden_size", type=int, default=256)
parser.add_argument("-graph_atten_dropout_prob", type=float, default=0.1)
parser.add_argument("-graph_dropout", type=float, default=0.1)

parser.add_argument("-num_heads", type=int, default=8)
parser.add_argument("-n_topic", type=int, default=100)
parser.add_argument("-hidden_dim", type=int, default=1024)
parser.add_argument("-N_enc_l", type=int, default=3)
parser.add_argument("-N_enc_g", type=int, default=3)
parser.add_argument("-N_dec", type=int, default=3)
parser.add_argument("-max_seq_len", type=int, default=100)
parser.add_argument("-max_doc_len", type=int, default=500)
parser.add_argument("-num_of_epochs", type=int, default=100)
parser.add_argument("-print_every", type=int, default=100)
parser.add_argument("-save_every", type=int, default=1000)
parser.add_argument("-validate_every",  type=int, default=1000)
parser.add_argument("-restore_old_checkpoint", type=bool, default=False)
parser.add_argument("-learning_rate", type=float, default=1e-4)
parser.add_argument("-learning_rate_NTM", type=float, default=5e-4)
parser.add_argument("-warmup_step",  type=int, default=1000)
parser.add_argument("-weight_decay", type=float, default=1e-6)
parser.add_argument("-dropout_rate", type=float, default=0.1)
parser.add_argument("-n_device", type=int, default=8)
parser.add_argument("-batch_size_per_device", type=int, default=1)
parser.add_argument("-max_extracted_sentences_per_document", type=int)
parser.add_argument("-moving_average_decay", type=float)
parser.add_argument("-p_stop_thres", type=float, default=0.7)
parser.add_argument("-apply_length_normalization", type=int, default=1)

args = parser.parse_args()

training_corpus_file_name = args.training_corpus_file_name
validation_corpus_file_name = args.validation_corpus_file_name
model_folder = args.model_folder
log_folder = args.log_folder
vocabulary_file_name = args.vocabulary_file_name
pretrained_unigram_embeddings_file_name = args.pretrained_unigram_embeddings_file_name
num_heads = args.num_heads
hidden_dim = args.hidden_dim
N_enc_l = args.N_enc_l
N_enc_g = args.N_enc_g
N_dec = args.N_dec
max_seq_len = args.max_seq_len
max_doc_len = args.max_doc_len  # 句子数量
num_of_epochs = args.num_of_epochs
print_every = args.print_every
save_every = args.save_every
validate_every = args.validate_every
restore_old_checkpoint = args.restore_old_checkpoint
learning_rate = args.learning_rate
learning_rate_NTM = args.learning_rate_NTM
warmup_step = args.warmup_step
weight_decay = args.weight_decay
dropout_rate = args.dropout_rate
n_device = args.n_device
batch_size_per_device = args.batch_size_per_device
max_extracted_sentences_per_document = args.max_extracted_sentences_per_document
moving_average_decay = args.moving_average_decay
p_stop_thres = args.p_stop_thres
n_topic = args.n_topic


if not os.path.exists(log_folder):
    os.makedirs(log_folder)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
log_out_file = log_folder + "/train.log"

training_corpus = load_corpus(training_corpus_file_name, True)
validation_corpus = load_corpus(validation_corpus_file_name, False)

with open(vocabulary_file_name, "rb") as f:
    words = pickle.load(f)
with open(pretrained_unigram_embeddings_file_name, "rb") as f:
    pretrained_embedding = pickle.load(f)
vocab = Vocab(words)
vocab_size, embed_dim = pretrained_embedding.shape

train_dataset = ExtractionTrainingDataset(training_corpus,  vocab, max_seq_len,  max_doc_len)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size_per_device * n_device, shuffle=True, num_workers=0, drop_last=True, worker_init_fn=lambda x: [np.random.seed(int(time.time())+x), torch.manual_seed(int(time.time()) + x)],  pin_memory=True)
total_number_of_samples = train_dataset.__len__()
val_dataset = ExtractionValidationDataset(validation_corpus, vocab, max_seq_len, max_doc_len)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size_per_device * n_device, shuffle=False, num_workers=0, drop_last=False, worker_init_fn=lambda x: [np.random.seed(int(time.time()) + 1 + x), torch.manual_seed(int(time.time()) + 1 + x)],  pin_memory=True)

local_sentence_encoder = LocalSentenceEncoder(vocab_size, vocab.pad_index, embed_dim, num_heads, hidden_dim, N_enc_l, pretrained_embedding)
global_context_encoder = GlobalContextEncoder(embed_dim, num_heads, hidden_dim, N_enc_g)
extraction_context_decoder = ExtractionContextDecoder(embed_dim, num_heads, hidden_dim, N_dec)
extractor = Extractor(embed_dim, num_heads)

topic = VAE([max_doc_len, int(max_doc_len / 2), int(max_doc_len / 4), n_topic], dropout_rate)
GAT_topic = GAT(embed_dim, embed_dim, args.graph_num_heads, args.graph_atten_dropout_prob)

# restore most recent checkpoint
if restore_old_checkpoint:
    ckpt = load_model(model_folder)
else:
    ckpt = None

if ckpt is not None:
    local_sentence_encoder.load_state_dict(ckpt["local_sentence_encoder"])
    global_context_encoder.load_state_dict(ckpt["global_context_encoder"])
    extraction_context_decoder.load_state_dict(ckpt["extraction_context_decoder"])
    extractor.load_state_dict(ckpt["extractor"])
    LOG("model restored!")
    print("model restored!")

gpu_list = np.arange(n_device).tolist()
device = torch.device("cuda:%d" % (gpu_list[0]) if torch.cuda.is_available() else "cpu")

local_sentence_encoder_ema = copy.deepcopy(local_sentence_encoder).to(device)
global_context_encoder_ema = copy.deepcopy(global_context_encoder).to(device)
extraction_context_decoder_ema = copy.deepcopy(extraction_context_decoder).to(device)
extractor_ema = copy.deepcopy(extractor).to(device)
topic_ema = copy.deepcopy(topic).to(device)
GAT_topic_ema = copy.deepcopy(GAT_topic).to(device)


local_sentence_encoder.to(device)
global_context_encoder.to(device)
extraction_context_decoder.to(device)
extractor.to(device)
topic.to(device)
GAT_topic.to(device)

if device.type == "cuda" and n_device > 1:
    local_sentence_encoder = nn.DataParallel(local_sentence_encoder, gpu_list)
    global_context_encoder = nn.DataParallel(global_context_encoder, gpu_list)
    extraction_context_decoder = nn.DataParallel(extraction_context_decoder, gpu_list)
    extractor = nn.DataParallel(extractor, gpu_list)
    topic = nn.DataParallel(topic, gpu_list)
    GAT_topic = nn.DataParallel(GAT_topic, gpu_list)

    local_sentence_encoder_ema = nn.DataParallel(local_sentence_encoder_ema, gpu_list)
    global_context_encoder_ema = nn.DataParallel(global_context_encoder_ema, gpu_list)
    extraction_context_decoder_ema = nn.DataParallel(extraction_context_decoder_ema, gpu_list)
    extractor_ema = nn.DataParallel(extractor_ema, gpu_list)
    topic_ema = nn.DataParallel(topic_ema, gpu_list)
    GAT_topic_ema = nn.DataParallel(GAT_topic_ema, gpu_list)

    model_parameters = [par for par in local_sentence_encoder.module.parameters() if par.requires_grad] + \
                       [par for par in global_context_encoder.module.parameters() if par.requires_grad] + \
                       [par for par in extraction_context_decoder.module.parameters() if par.requires_grad] + \
                       [par for par in extractor.module.parameters() if par.requires_grad]

    model_parameters_ = [par for par in topic.module.parameters() if par.requires_grad] + \
                        [par for par in GAT_topic.module.parameters() if par.requires_grad]

else:
    model_parameters = [par for par in local_sentence_encoder.parameters() if par.requires_grad] + \
                    [par for par in global_context_encoder.parameters() if par.requires_grad] + \
                    [par for par in extraction_context_decoder.parameters() if par.requires_grad] + \
                    [par for par in extractor.parameters() if par.requires_grad]

    model_parameters_ = [par for par in topic.parameters() if par.requires_grad] + \
                        [par for par in GAT_topic.parameters() if par.requires_grad]

optimizer = torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)

optimizer_NTM = torch.optim.Adam(model_parameters_, lr=learning_rate_NTM, weight_decay=weight_decay)

optim = [optimizer, optimizer_NTM]

if ckpt is not None:
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        LOG("optimizer restored!")
        print("optimizer restored!")
    except:
        pass

current_epoch = 0
current_batch = 0

if ckpt is not None:
    current_batch = ckpt["current_batch"]
    current_epoch = int(current_batch * batch_size_per_device * n_device / total_number_of_samples)
    LOG("current_batch restored!")
    print("current_batch restored!")

np.random.seed()

rouge_cal = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def train_iteration(batch):
    seqs, doc_mask, selected_y_label, selected_score, valid_sen_idxs = batch
    seqs = seqs.to(device)
    doc_mask = doc_mask.to(device)
    selected_y_label = selected_y_label.to(device)
    selected_score = selected_score.to(device)
    valid_sen_idxs = valid_sen_idxs.to(device)

    num_documents = seqs.size(0)
    num_sentences = seqs.size(1)
    
    local_sen_embed = local_sentence_encoder(seqs.view(-1, seqs.size(2)), dropout_rate)
    local_sen_embed = local_sen_embed.view(-1, num_sentences, local_sen_embed.size(1))
    global_context_embed = global_context_encoder(local_sen_embed, doc_mask, dropout_rate)
    
    doc_mask_np = doc_mask.detach().cpu().numpy()
    remaining_mask_np = np.ones_like(doc_mask_np).astype(bool) | doc_mask_np
    extraction_mask_np = np.zeros_like(doc_mask_np).astype(bool) | doc_mask_np
    
    log_action_prob_list = []
    log_stop_prob_list = []
    loss_top_list = []
    
    done_list = []
    extraction_context_embed = None
    redundancy_embed = None

    for step in range(valid_sen_idxs.shape[1]):
        remaining_mask = torch.from_numpy(remaining_mask_np).to(device)
        extraction_mask = torch.from_numpy(extraction_mask_np).to(device)

        # 根据当前的句子生成文章主题
        topic_embed, loss_top = topic(local_sen_embed, extraction_mask, dropout_rate, redundancy_embed)

        # 根据主题修改句子的上下文编码
        global_context_embed, topic_embed = GAT_topic(topic_embed, global_context_embed, extraction_mask)

        redundancy_embed = topic_embed[:, 0]

        if step > 0:
            extraction_context_embed = extraction_context_decoder(local_sen_embed, remaining_mask, extraction_mask, dropout_rate)  # 每次筛选完句子之后组成的文本编码
        p, p_stop = extractor(local_sen_embed, global_context_embed, topic_embed, extraction_context_embed, extraction_mask, dropout_rate)

        p_stop = p_stop.unsqueeze(1)
        m_stop = Categorical(torch.cat([1-p_stop, p_stop], dim=1))
        
        sen_indices = valid_sen_idxs[:, step]
        done = sen_indices == -1
        if len(done_list) > 0:
            # 逻辑或
            done = torch.logical_or(done_list[-1], done)
            just_stop = torch.logical_and(~done_list[-1], done)
        else:
            just_stop = done
        
        if torch.all(done) and not torch.any(just_stop):
            break
        # 每个句子的分数
        p = p.masked_fill(extraction_mask, 1e-12)
        normalized_p = p / p.sum(dim=1, keepdims=True)

        normalized_p = normalized_p[np.arange(num_documents), sen_indices]  # 每次应该选择的句子的分数
        log_action_prob = normalized_p.masked_fill(done, 1.0).log()
        loss_top = loss_top.masked_fill(done, 0.0)

        log_stop_prob = m_stop.log_prob(done.to(torch.long))
        log_stop_prob = log_stop_prob.masked_fill(torch.logical_xor(done, just_stop), 0.0)
        
        log_action_prob_list.append(log_action_prob.unsqueeze(1))
        log_stop_prob_list.append(log_stop_prob.unsqueeze(1))
        loss_top_list.append(loss_top.unsqueeze(0))
        done_list.append(done)
        
        for doc_i in range(num_documents):
            sen_i = sen_indices[doc_i].item()
            if sen_i != -1:
                remaining_mask_np[doc_i, sen_i] = False
                extraction_mask_np[doc_i, sen_i] = True

    log_action_prob_list = torch.cat(log_action_prob_list, dim=1)
    log_stop_prob_list = torch.cat(log_stop_prob_list, dim=1)
    loss_top_list = torch.cat(loss_top_list, dim=0).T
    log_prob_list = -(log_action_prob_list + log_stop_prob_list) + loss_top_list

    if args.apply_length_normalization:
        log_prob_list = log_prob_list.sum(dim=1) / ((log_prob_list != 0).to(torch.float32).sum(dim=1))
    else:
        log_prob_list = log_prob_list.sum(dim=1) 

    loss = (log_prob_list * selected_score).mean()

    for o in optim:
        o.zero_grad()

    loss.backward()

    for o in optim:
        o.step()

    return loss.item()


def validation_iteration(batch):
    seqs, doc_mask, sentences, summary = batch
    seqs = seqs.to(device)
    doc_mask = doc_mask.to(device)

    num_sentences = seqs.size(1)
    local_sen_embed = local_sentence_encoder_ema(seqs.view(-1, seqs.size(2)))
    local_sen_embed = local_sen_embed.view(-1, num_sentences, local_sen_embed.size(1))
    global_context_embed = global_context_encoder_ema(local_sen_embed, doc_mask)
    
    num_documents = seqs.size(0)
    doc_mask = doc_mask.detach().cpu().numpy()
    remaining_mask_np = np.ones_like(doc_mask).astype(np.bool) | doc_mask
    extraction_mask_np = np.zeros_like(doc_mask).astype(np.bool) | doc_mask
    
    
    done_list = []
    extraction_context_embed = None
    redundancy_embed = None

    for step in range(max_extracted_sentences_per_document):
        remaining_mask = torch.from_numpy(remaining_mask_np).to(device)
        extraction_mask = torch.from_numpy(extraction_mask_np).to(device)

        # 根据当前的句子生成文章主题
        topic_embed, _ = topic(local_sen_embed, extraction_mask, dropout_rate, redundancy_embed)

        # 根据主题修改句子的上下文编码
        global_context_embed, topic_embed = GAT_topic(topic_embed, global_context_embed, extraction_mask)

        redundancy_embed = topic_embed[:, 0]

        if step > 0:
            extraction_context_embed = extraction_context_decoder(local_sen_embed, remaining_mask, extraction_mask,
                                                                  dropout_rate)  # 每次筛选完句子之后组成的文本编码

        p, p_stop = extractor(local_sen_embed, global_context_embed, topic_embed, extraction_context_embed,
                      extraction_mask, dropout_rate)

        p = p.masked_fill(extraction_mask, 1e-12)
        normalized_p = p / (p.sum(dim=1, keepdims=True))

        stop_action = p_stop > p_stop_thres

        done = stop_action | torch.all(extraction_mask, dim=1)
        if len(done_list) > 0:
            done = torch.logical_or(done_list[-1], done)
        if torch.all(done):
            break
            
        sen_indices = torch.argmax(normalized_p, dim=1)
        done_list.append(done)
        
        for doc_i in range(num_documents):
            if not done[doc_i]:
                sen_i = sen_indices[doc_i].item()
                remaining_mask_np[doc_i, sen_i] = False
                extraction_mask_np[doc_i, sen_i] = True
                
    sentences = list(zip(* sentences))
    summary = list(zip(*summary))
                
    scores = []
    for doc_i in range(len(sentences)):
        ref = "\n".join(summary[doc_i]).strip()
        extracted_sen_indices = np.argwhere(remaining_mask_np[doc_i] == False)[:, 0]
        hyp = "\n".join([sentences[doc_i][idx] for idx in extracted_sen_indices]).strip()

        score = rouge_cal.score(hyp, ref)
        scores.append((score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure))

    return scores


for epoch in range(current_epoch, num_of_epochs):
    running_loss = 0 
    
    for count, batch in tqdm(enumerate(train_data_loader)):
        loss = train_iteration(batch)
        running_loss += loss

        update_moving_average(local_sentence_encoder_ema,  local_sentence_encoder, moving_average_decay)
        update_moving_average(global_context_encoder_ema,  global_context_encoder, moving_average_decay)
        update_moving_average(extraction_context_decoder_ema,  extraction_context_decoder, moving_average_decay)
        update_moving_average(extractor_ema,  extractor, moving_average_decay)
        
        current_batch += 1
        if current_batch % print_every == 0:
            current_learning_rate = get_lr(optimizer)[0]
            LOG("[current_batch: %05d] loss: %.3f, learning rate: %f" % (current_batch, running_loss/print_every,  current_learning_rate))
            print("[current_batch: %05d] loss: %.3f, learning rate: %f" % (current_batch, running_loss/print_every, current_learning_rate))
            os.system("nvidia-smi > %s/gpu_usage.log"%( log_folder ))
            running_loss = 0

            if current_learning_rate < 1e-6:
                print("No progress is being made, stop training!")
                sys.exit(0)
        
        if validate_every != 0 and current_batch % validate_every == 0:
            print("Starting validation ...")
            LOG("Starting validation ...")
            # validation
            val_score_list = []
            with torch.no_grad():
                for batch in tqdm(val_data_loader):
                    val_score_list += validation_iteration(batch)

            val_rouge1, val_rouge2, val_rougeL = list(zip(*val_score_list))

            avg_val_rouge1 = np.mean(val_rouge1)
            avg_val_rouge2 = np.mean(val_rouge2)
            avg_val_rougeL = np.mean(val_rougeL)
            print("val: %.4f, %.4f, %.4f" % (avg_val_rouge1, avg_val_rouge2, avg_val_rougeL))
            LOG("val: %.4f, %.4f, %.4f" % (avg_val_rouge1, avg_val_rouge2, avg_val_rougeL))

        if current_batch % save_every == 0:
            save_model({
                "current_batch": current_batch,
                "local_sentence_encoder": local_sentence_encoder_ema,
                "global_context_encoder": global_context_encoder_ema,
                "extraction_context_decoder": extraction_context_decoder_ema,
                "topic": topic_ema,
                "GAT_topic": GAT_topic_ema,
                "extractor": extractor_ema,
                "optimizer": optimizer.state_dict()
                }, model_folder+"/model_batch_%d.pt" % (current_batch), max_to_keep=100)


