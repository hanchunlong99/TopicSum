# _*_ coding: utf-8 _*_
# @Time: 2022/10/10 9:33 
# @Author: 韩春龙
from summarizers import MemSum
from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import numpy as np

rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

# memsum_pubmed = MemSum("../model/MemSum_Full/pubmed/200dim/run0/model_batch_65000.pt",
#                   "../model/glove/vocabulary_200dim.pkl",
#                   gpu = 0 ,  max_doc_len = 500  )
#
# memsum_pubmed_truncated = MemSum(  "../model/MemSum_Full/pubmed_truncated/200dim/run0/model_batch_49000.pt",
#                   "../model/glove/vocabulary_200dim.pkl",
#                   gpu = 0 ,  max_doc_len = 50  )
#
# memsum_arxiv = MemSum(  "../model/MemSum_Full/arxiv/200dim/run0/model_batch_37000.pt",
#                   "../model/glove/vocabulary_200dim.pkl",
#                   gpu = 0,  max_doc_len=500)

memsum_gov_report = MemSum("model/MemSum_Full/gov-report/model_batch_34000.pt", "model/glove/vocabulary_200dim.pkl", gpu=0,  max_doc_len=500)

# test_corpus_pubmed = [ json.loads(line) for line in open("../data/pubmed/test_PUBMED.jsonl") ]
# test_corpus_pubmed_truncated = [ json.loads(line) for line in open("../data/pubmed_truncated/test_PUBMED.jsonl") ]
# test_corpus_arxiv = [ json.loads(line) for line in open("../data/arxiv/test_ARXIV.jsonl") ]
test_corpus_gov_report = [json.loads(line) for line in open("data/gov-report/test_GOVREPORT.jsonl")]


def evaluate(model, corpus, p_stop, max_extracted_sentences, rouge_cal):

    val_score_list = []
    for data in tqdm(corpus):
        sentences = data["text"]
        summary = data["summary"]
        sentences = sentences[:500]
        num_sentences_in_doc = len(sentences)

        sentences += [""] * (500 - num_sentences_in_doc)

        summary = summary[:500]
        if len(summary) < 500:
            summary = summary + [""] * (500 - len(summary))

        val_score_list += model.extract([data["text"]], p_stop_thres=p_stop,
                                          max_extracted_sentences_per_document=max_extracted_sentences, sentences=sentences, summary=summary)

    val_rouge1, val_rouge2, val_rougeL = list(zip(*val_score_list))

    avg_val_rouge1 = np.mean(val_rouge1)
    avg_val_rouge2 = np.mean(val_rouge2)
    avg_val_rougeL = np.mean(val_rougeL)
    print("val: %.4f, %.4f, %.4f" % (avg_val_rouge1, avg_val_rouge2, avg_val_rougeL))


evaluate(memsum_gov_report, test_corpus_gov_report, 0.6, 22, rouge_cal)
