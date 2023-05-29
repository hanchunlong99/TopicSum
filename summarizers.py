from src.MemSum_Full.model import LocalSentenceEncoder as LocalSentenceEncoder_MemSum_Full
from src.MemSum_Full.model import GlobalContextEncoder as GlobalContextEncoder_MemSum_Full
from src.MemSum_Full.model import ExtractionContextDecoder as ExtractionContextDecoder_MemSum_Full
from src.MemSum_Full.model import Extractor as Extractor_MemSum_Full
from src.MemSum_Full.model import VAE as VAE
from src.MemSum_Full.model import GAT as GAT


from src.MemSum_Full.datautils import Vocab as Vocab_MemSum_Full
from src.MemSum_Full.datautils import SentenceTokenizer as SentenceTokenizer_MemSum_Full
from rouge_score import rouge_scorer

import torch.nn.functional as F
from torch.distributions import Categorical

import pickle
import torch
import numpy as np

from tqdm import tqdm
import json

rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)
class MemSum:
    def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=200, num_heads=8, hidden_dim = 1024, N_enc_l = 2 , N_enc_g = 2, N_dec = 3,  max_seq_len =100, max_doc_len = 500  ):
        with open( vocabulary_path , "rb" ) as f:
            words = pickle.load(f)
        self.vocab = Vocab_MemSum_Full( words )
        vocab_size = len(words)
        self.local_sentence_encoder = LocalSentenceEncoder_MemSum_Full( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.global_context_encoder = GlobalContextEncoder_MemSum_Full( embed_dim, num_heads, hidden_dim, N_enc_g )
        self.extraction_context_decoder = ExtractionContextDecoder_MemSum_Full( embed_dim, num_heads, hidden_dim, N_dec )
        self.extractor = Extractor_MemSum_Full( embed_dim, num_heads )
        self.topic = VAE([max_doc_len, int(max_doc_len / 2), int(max_doc_len / 4), 50], 0.1)
        self.GAT_topic = GAT(embed_dim, embed_dim, 8, 0.1)

        ckpt = torch.load(model_path, map_location="cpu")
        self.local_sentence_encoder.load_state_dict(ckpt["local_sentence_encoder"])
        self.global_context_encoder.load_state_dict(ckpt["global_context_encoder"])
        self.extraction_context_decoder.load_state_dict(ckpt["extraction_context_decoder"])
        self.extractor.load_state_dict(ckpt["extractor"])
        self.topic.load_state_dict(ckpt["topic"])
        self.GAT_topic.load_state_dict(ckpt["GAT_topic"])

        self.device = torch.device("cuda:%d"%(gpu) if gpu is not None and torch.cuda.is_available() else "cpu")
        self.local_sentence_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        self.topic.to(self.device)
        self.GAT_topic.to(self.device)

        self.sentence_tokenizer = SentenceTokenizer_MemSum_Full()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
    
    def get_ngram(self,  w_list, n = 4 ):
        ngram_set = set()
        for pos in range(len(w_list) - n + 1 ):
            ngram_set.add( "_".join( w_list[ pos:pos+n] )  )
        return ngram_set

    def extract( self, document_batch, p_stop_thres = 0.7, ngram_blocking = False, ngram = 3,
                 return_sentence_position = False, return_sentence_score_history = False,
                 max_extracted_sentences_per_document = 4, sentences=None, summary=None):
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize(sen)
                tokenized_document.append(tokenized_sen)
                sentence_length_list.append(len(tokenized_sen.split()))
            tokenized_document_batch.append(tokenized_document)
            document_length_list.append(len(tokenized_document))

        max_document_length = self.max_doc_len
        max_sentence_length = self.max_seq_len
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                document = document[:max_document_length]
            else:
                document = document + [""] * (max_document_length - len(document))

            doc_mask.append([1 if sen.strip() == "" else 0 for sen in document])

            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq(sen, max_sentence_length)
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs, dtype=np.int64)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed = self.local_sentence_encoder(seqs.view(-1, seqs.size(2)))
            sen_embed = sen_embed.view(-1, num_sentences, sen_embed.size(1))
            relevance_embed = self.global_context_encoder(sen_embed, doc_mask)
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(np.bool) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(np.bool) | current_doc_mask
        
                current_sen_embed = sen_embed[doc_i:doc_i+1]  # 局部
                current_relevance_embed = relevance_embed[ doc_i:doc_i+1 ]  # 全局
                current_redundancy_embed = None

                redundancy_embed = None
                done_list = []
                for step in range(max_extracted_sentences_per_document+1):
                    current_remaining_mask = torch.from_numpy(current_remaining_mask_np).to(self.device)
                    current_extraction_mask = torch.from_numpy(current_extraction_mask_np).to(self.device)

                    topic_embed, loss_top = self.topic(current_sen_embed, current_extraction_mask, 0.1, redundancy_embed)
                    current_relevance_embed, topic_embed = self.GAT_topic(topic_embed, current_relevance_embed, current_extraction_mask)

                    redundancy_embed = topic_embed[:, 0]
                    if step > 0:
                        current_redundancy_embed = self.extraction_context_decoder(current_sen_embed, current_remaining_mask, current_extraction_mask, 0.1)
                    p, p_stop = self.extractor(current_sen_embed, current_relevance_embed, topic_embed, current_redundancy_embed, current_extraction_mask, 0.1)


                    # p_stop = p_stop.unsqueeze(1)

                    p = p.masked_fill(current_extraction_mask, 1e-12)

                    normalized_p = p / p.sum(dim=1, keepdims=True)

                    stop_action = p_stop > p_stop_thres

                    done = stop_action | torch.all(current_extraction_mask, dim=1)  # 后半部分为TRUE的话意味着所有句子全部选完了
                    # done = torch.all(extraction_mask, dim=1)    # 后半部分为TRUE的话意味着所有句子全部选完了
                    if len(done_list) > 0:
                        done = torch.logical_or(done_list[-1], done)  # 逻辑或
                    if torch.all(done):
                        break

                    sen_indices = torch.argmax(normalized_p, dim=1)
                    done_list.append(done)

                    for doc_i in range(num_documents):
                        if not done[doc_i]:
                            sen_i = sen_indices[doc_i].item()
                            current_remaining_mask_np[doc_i, sen_i] = False
                            current_extraction_mask_np[doc_i, sen_i] = True

                scores = []

                ref = "\n".join(summary).strip()
                extracted_sen_indices = np.argwhere(current_remaining_mask_np[0] == False)[:, 0]
                hyp = "\n".join([sentences[idx] for idx in extracted_sen_indices]).strip()

                score = rouge_cal.score(hyp, ref)
                scores.append((score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure))

                return scores



