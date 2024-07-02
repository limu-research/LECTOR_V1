import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,BertModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from transformers import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import NMF
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
import re
import regex
import nagisa
import umap
import hdbscan
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("BERT_models/bert-large-japanese")
model = BertModel.from_pretrained('BERT_models/bert-finetuned-PROGRAMING_RON_new_v2',output_hidden_states = True, output_attentions=True)
softmax_row = torch.nn.Softmax(dim=1)
softmax_column = torch.nn.Softmax(dim=0)
scaler_cross = StandardScaler()
scaler_self = StandardScaler()
scaler_final = StandardScaler()

class LECTOR_SLIDE_Pipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
                  
        # ENABLE THE USE OF ARGUMENT TITLE_EMBEDD FOR POSTPROCESSING
        postprocess_kwargs = {}
        if "params" in kwargs:
            postprocess_kwargs["params"] = kwargs["params"]
        
#         # ENABLE THE USE OF ARGUMENT N_LAYERS FOR POSTPROCESSING
#         postprocess_kwargs = {}
#         if "n_layers" in kwargs:
#             postprocess_kwargs["n_layers"] = kwargs["n_layers"]
            
#         # ENABLE THE USE OF ARGUMENT PSI FOR POSTPROCESSING
#         postprocess_kwargs = {}
#         if "n_layers" in kwargs:
#             postprocess_kwargs["psi"] = kwargs["psi"]
            
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, inputs, maybe_arg=2):
        # SPLIT THE TEXT INTO TITLE AND BODY TEXT
        s_slide = inputs.split()
        title = s_slide[0]
        title = re.sub(r"|[*]|[+]|[;]|[#]|[/]|[0-9]","",title)
        body = '。'.join(s_slide[1:])
        body = re.sub(r"|[+]|[;]|[#]|[/]|[0-9]","",body)
        
        # TOKENIZE THE INPUT
        tokens = self.tokenizer.tokenize("[CLS] " + title + ":" + body + " [SEP]")
        tokens_title = self.tokenizer.tokenize("[CLS] " + title + " [SEP]")
        tokens_body = self.tokenizer.tokenize("[CLS] " + body + " [SEP]")
    
        # OBTAIN THE TOKEN_IDS TENSOR
        if(len(tokens)<=512):
            tokens_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        else:
            # TRIM THE LIST IF IS LONGER THAN 512 TOKENS
            indexes_aux = self.tokenizer.convert_tokens_to_ids(tokens)[:511]
            indexes_aux.append(3)
            tokens_aux = tokens[:511]
            tokens_aux.append("[SEP]")
            tokens = tokens_aux[:]
            tokens_tensor = torch.tensor([indexes_aux])
        
        tokens_tensor_title = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens_title)])
        
        if(len(tokens_body)<=512):
            tokens_tensor_body = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens_body)])
        else:
            # TRIM THE LIST IF IS LONGER THAN 512 TOKENS
            body_indexes_aux = self.tokenizer.convert_tokens_to_ids(tokens_body)[:511]
            body_indexes_aux.append(3)
            body_tokens_aux = tokens_body[:511]
            body_tokens_aux.append("[SEP]")
            tokens_body = body_tokens_aux[:]
            tokens_tensor_body = torch.tensor([body_indexes_aux])
            
        # RETURN THE TENSOR INPUT FOR THE MODEL AND THE TOKENS
        for_processing = {"model_input":{"input_ids": tokens_tensor},
                          "model_input_title":{"input_ids": tokens_tensor_title},
                          "model_input_body":{"input_ids": tokens_tensor_body},
                          "tokens":tokens,
                          "tokens_title":tokens_title,
                          "tokens_body":tokens_body}
        return for_processing

    def _forward(self, model_inputs):
        # RETRIEVE THE VALUES FROM THE PREPROCESS METHOD
        model_input = model_inputs["model_input"]
        model_input_title = model_inputs["model_input_title"]
        model_input_body = model_inputs["model_input_body"]
        
        # PROCESS THE TENSOR INPUT WITH THE BERT MODEL
        output = self.model(**model_input)
        output_title = self.model(**model_input_title)
        output_body = self.model(**model_input_body)
        
        # RETURN THE MODEL OUTPUTS AND THE TOKENS
        for_postprocessing = {"output":output,
                              "output_title":output_title,
                              "output_body":output_body,
                              "tokens":model_inputs["tokens"],
                              "tokens_title":model_inputs["tokens_title"],
                              "tokens_body":model_inputs["tokens_body"]}
        return for_postprocessing

    def postprocess(self, model_outputs, params={"title_embedd":torch.tensor([1]),"n_layers":12,"psi_t":0.5,"psi_b":0.5}):   
        # GET THE PARAMETER VALUES
        E_T = params["title_embedd"]
        n_layers = params["n_layers"]
        psi_t = params["psi_t"]
        psi_b = params["psi_b"]
#         print(params)
        
        # RETRIEVE THE VALUES FROM THE FORWARD METHOD
        output = model_outputs["output"]
        output_title = model_outputs["output_title"]
        output_body = model_outputs["output_body"]
        tokens = model_outputs["tokens"]
        tokens_title = model_outputs["tokens_title"]
        tokens_body = model_outputs["tokens_body"]
        
        # ESTIMATE A SINGLE ATTENTION MAP
        attention = output[3]
        attention = torch.stack(attention, dim=0)
        attention = torch.squeeze(attention, dim=1)
        attention = torch.mean(attention,1)
        attention = torch.mean(attention[-12:],0)
        
        # ESTIMATE A SINGLE EMBEDDING PER TOKEN
        embeddings = output[2]
        embeddings = torch.stack(embeddings, dim=0)
        embeddings = torch.squeeze(embeddings, dim=1)
        embeddings = torch.cat((list(embeddings[-n_layers:])),1)
        
        embeddings_title = output_title[2]
        embeddings_title = torch.stack(embeddings_title, dim=0)
        embeddings_title = torch.squeeze(embeddings_title, dim=1)
        embeddings_title = torch.cat((list(embeddings_title[-n_layers:])),1)
        
        embeddings_body = output_body[2]
        embeddings_body = torch.stack(embeddings_body, dim=0)
        embeddings_body = torch.squeeze(embeddings_body, dim=1)
        embeddings_body = torch.cat((list(embeddings_body[-n_layers:])),1)
        
        # CORRECT THE TOKENS (WORD-WISE)
        n_embeddings, n_attention, n_tokens = self.correct_tokens(embeddings, attention, tokens)
        nt_embeddings, nt_tokens = self.correct_embeddings(embeddings_title, tokens_title)
        nb_embeddings, nb_tokens = self.correct_embeddings(embeddings_body, tokens_body)
        
        # ESTIMATE THE WORD IMPORTANCE FROM THE ATTENTION MAPS
        n_attention.fill_diagonal_(0)
        word_importance = torch.sum(n_attention, 0)
        word_importance = word_importance[1:len(n_tokens)-1]
        
        # DROP [CLS] AND [SEP] TOKENS
        n_tokens = n_tokens[1:-1]
        nt_tokens = nt_tokens[1:-1]
        nb_tokens = nb_tokens[1:-1]
        
        n_embeddings = n_embeddings[1:-1]
        nt_embeddings = nt_embeddings[1:-1]
        nb_embeddings = nb_embeddings[1:-1]
    
        # CORRECT THE TITLE EMBEDDINGS
        nt_tokens, nt_embeddings = self.extract_title(nt_tokens,nt_embeddings)
        
        # PREPARE SLIDE SINGLE REPRESENTATION BASED ON THE DISCOURSE
        try:
            E_lt = nt_embeddings
            E_lb = nb_embeddings
        
            fail_1 = (len(nt_embeddings.numpy().tolist())==0)
            fail_2 = (len(nb_embeddings.numpy().tolist())==0)
        
            if (not fail_2) and (not fail_1):
                #####################################################
                S_1 = self.self_attention(E_T,E_lt,psi_t,n_layers)
                aux_1 = softmax_row(S_1)
                aux_2 = torch.unsqueeze(torch.mean(softmax_column(S_1),1),1)
                # aux_3 = AVE_row(SOF_col(S_1))SOF_row(S_1)
                aux_3 = torch.matmul(aux_2.T,aux_1)
                p_stitle = torch.matmul(aux_3,E_lt)
                ######################################################
                S_2 = self.self_attention(E_lt,E_lb,psi_b,n_layers)
                aux_4 = softmax_row(S_2)
                # aux_5 = Pr(w e st_i|Est_1)Pr(w e sb_i|Est_i)
                aux_5 = torch.matmul(aux_3,aux_4)
                ######################################################
                p_sbody = torch.matmul(aux_5,E_lb)
            elif (not fail_2) and fail_1:
                ######################################################
                S_2 = self.self_attention(E_T,E_lb,psi_b,n_layers)
                aux_1 = softmax_row(S_2)
                aux_2 = torch.unsqueeze(torch.mean(softmax_column(S_2),1),1)
                aux_3 = torch.matmul(aux_2.T,aux_1)
                p_sbody = torch.matmul(aux_3,E_lb)
                ######################################################
                p_stitle = p_sbody
            elif fail_2 and (not fail_1):
                #####################################################
                S_1 = self.self_attention(E_T,E_lt,psi_t,n_layers)
                aux_1 = softmax_row(S_1)
                aux_2 = torch.unsqueeze(torch.mean(softmax_column(S_1),1),1)
                aux_3 = torch.matmul(aux_2.T,aux_1)
                p_stitle = torch.matmul(aux_3,E_lt)
                ######################################################
                p_sbody = p_stitle
            elif fail_2 and fail_1:
                p_stitle = torch.zeros(1,n_layers*1024)
                p_sbody = torch.zeros(1,n_layers*1024)
        
        except ValueError:
            print("Main Title embedding format is incorrect.")
        
        
        # PREPARE CANDIDATES DATAFRAME
        candidates, a_ij, E_jt = self.extract_candidates(n_tokens,word_importance,n_embeddings)
        
        candidates ={"candidate":candidates,"embeddings":E_jt,"importance":a_ij}
        
        # RETURN THE ESTIMATED ATTENTION MAP, EMBEDDINGS AND THE TOKENS
        output_dict = {"candidates_info":candidates,
                       "title_tokens":nt_tokens,
                       "body_tokens":nb_tokens,
                       "embeddings_title":p_stitle,
                       "embeddings_body":p_sbody}
        return output_dict
    
    def correct_tokens(self, prev_embeddings, prev_attention, prev_tokens):
        # AUXILIARY VARIABLES
        i = 0
        j = 1
        aux_o = 0
        aux_f = 0
        aux_bool = False
        test_list = []
        n_repeats = 0
        new_tokens = []
        new_indices = []
        indices_len = []
        
        # MAIN LOOP FOR CORRECTING TOKENS
        for token in prev_tokens:
            # WHETHER THE TOKEN STARTS WITH AN "##"
            if not(token == token.replace("##", "")):
                j += 1
                n_repeats += 1
                # WHETHER IS A NEW WORD
                if(aux_bool):
                    test_list.append(i)
                    indices_len.pop(-1)
                    indices_len.append(j)
                else:
                    i -= 1
                    test_list.append(i)
                    indices_len.append(j)
                    new_indices.append(i)
                # UPDATE THE STATE
                aux_t = new_tokens[-1]
                new_tokens.pop(-1)
                new_tokens.append(aux_t + token.replace("##", ""))
                aux_bool = True
            else:
                if(aux_bool):
                    i += 1
                aux_bool = False
                test_list.append(i)
                new_tokens.append(token)
                j = 1
                i += 1
        if len(new_indices)>=1:
            # CORRECT THE EMBEDDINGS
            index = torch.tensor(test_list).repeat(prev_embeddings.shape[1],1).T
            new_embeddings = torch.zeros(prev_embeddings.shape[0]-n_repeats, prev_embeddings.shape[1], dtype=prev_embeddings.dtype).scatter_(0,index,prev_embeddings, reduce='add')
            
            # CORRECT THE ATTENTION MAP
            index_1 = torch.tensor(test_list).repeat(prev_attention.shape[0],1)
            index_2 = torch.tensor(test_list).repeat(prev_attention.shape[0]-n_repeats,1).T
            attention_fc = torch.zeros(prev_attention.shape[0], prev_attention.shape[0]-n_repeats, dtype=prev_attention.dtype).scatter_(1,index_1,prev_attention, reduce='add')
            new_attention = torch.zeros(prev_attention.shape[0]-n_repeats,prev_attention.shape[0]-n_repeats, dtype=prev_attention.dtype).scatter_(0,index_2,attention_fc, reduce='add')
            
            for i_index in range(len(new_indices)):
                n_index = new_indices[i_index]
                n_len = indices_len[i_index]
                new_attention[n_index]/=n_len
                new_embeddings[n_index]/=n_len
        else:
            # CORRECTION IS NOT REQUIRED
            new_attention = prev_attention
            new_embeddings = prev_embeddings
            new_tokens = prev_tokens
        
        # RETURN THE CORRECTED VALUES
        return new_embeddings, new_attention,new_tokens
    
    def correct_embeddings(self, prev_embeddings, prev_tokens):
        # AUXILIARY VARIABLES
        i = 0
        j = 1
        aux_o = 0
        aux_f = 0
        aux_bool = False
        test_list = []
        n_repeats = 0
        new_tokens = []
        new_indices = []
        indices_len = []
        
        # MAIN LOOP FOR CORRECTING TOKENS
        for token in prev_tokens:
            # WHETHER THE TOKEN STARTS WITH AN "##"
            if not(token == token.replace("##", "")):
                j += 1
                n_repeats += 1
                # WHETHER IS A NEW WORD
                if(aux_bool):
                    test_list.append(i)
                    indices_len.pop(-1)
                    indices_len.append(j)
                else:
                    i -= 1
                    test_list.append(i)
                    indices_len.append(j)
                    new_indices.append(i)
                # UPDATE THE STATE
                aux_t = new_tokens[-1]
                new_tokens.pop(-1)
                new_tokens.append(aux_t + token.replace("##", ""))
                aux_bool = True
            else:
                if(aux_bool):
                    i += 1
                aux_bool = False
                test_list.append(i)
                new_tokens.append(token)
                j = 1
                i += 1
        if len(new_indices)>=1:           
            # CORRECT THE EMBEDDINGS
            index = torch.tensor(test_list).repeat(prev_embeddings.shape[1],1).T
            new_embeddings = torch.zeros(prev_embeddings.shape[0]-n_repeats, prev_embeddings.shape[1], dtype=prev_embeddings.dtype).scatter_(0,index,prev_embeddings, reduce='add')
            for i_index in range(len(new_indices)):
                n_index = new_indices[i_index]
                n_len = indices_len[i_index]
                new_embeddings[n_index]/=n_len
        else:
            # CORRECTION IS NOT REQUIRED
            new_embeddings = prev_embeddings
            new_tokens = prev_tokens
        
        # RETURN THE CORRECTED VALUES
        return new_embeddings,new_tokens
    
    def extract_candidates(self,slide_words,words_importance,words_embeddings):
#         print(slide_words)
        # AUXILIARY VARIABLES
        words_importance = words_importance.numpy()
        phrases = []
        attentions = []
        embeddings = []
        n_words = len(slide_words)
        # LOOP THROUGH ALL WORDS
        for i in range(n_words-1):
            mini_doc_1 = nagisa.tagging(slide_words[i])
            mini_doc_2 = nagisa.tagging(slide_words[i+1])
            len_cond_1 = (len(mini_doc_1.postags)<3) and (len(slide_words[i])<10)
            len_cond_2 = (len(mini_doc_2.postags)<3) and (len(slide_words[i+1])<10)
            if ("名詞" in mini_doc_1.postags)and(len_cond_1):
                phrases.append(slide_words[i])
                attentions.append(words_importance[i])
                embeddings.append(torch.unsqueeze(words_embeddings[i],0))
                if ("名詞" in mini_doc_2.postags)and(len_cond_2):
                    phrases.append(slide_words[i]+
                                   slide_words[i+1])
                    attentions.append(words_importance[i]+words_importance[i+1])
                    auxiliar_embedd = torch.unsqueeze((words_embeddings[i] + words_embeddings[i+1])/2,1)
                    embeddings.append(auxiliar_embedd.T)
        mini_doc_1 = nagisa.tagging(slide_words[n_words-1])
        len_cond_1 = (len(mini_doc_1.postags)<3) and (len(slide_words[n_words-1])<10)
        if ("名詞" in mini_doc_1.postags)and(len_cond_1):
            phrases.append(slide_words[n_words-1])
            attentions.append(words_importance[n_words-1])
            embeddings.append(torch.unsqueeze(words_embeddings[n_words-1],0))
#         print(phrases)
        embeddings_t = torch.cat(embeddings,axis=0)
        return phrases,attentions,embeddings_t
    
    def extract_title(self,title_tokens,title_embeddings):
        # AUXILIARY VARIABLES
        title_words = []
        embeddings = []
        n_words = len(title_tokens)
        # LOOP THROUGH ALL WORDS
        for i in range(n_words):
            mini_doc = nagisa.tagging(title_tokens[i])
            if ("名詞" in mini_doc.postags):
                title_words.append(title_tokens[i])
                embeddings.append(torch.unsqueeze(title_embeddings[i],0))
        if not len(embeddings)==0:
            embeddings_t = torch.cat(embeddings,axis=0)
        else:
            embeddings_t = torch.tensor([])
        return title_words,embeddings_t
    
    def self_attention(self,E_A,E_B,psi,n):
        S = torch.matmul(E_A,E_B.T)/(psi*1024)**0.5
        return S

class LECTOR_TITLE_Pipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        
        # ENABLE THE USE OF ARGUMENT N_LAYERS FOR POSTPROCESSING
        postprocess_kwargs = {}
        if "n_layers" in kwargs:
            postprocess_kwargs["n_layers"] = kwargs["n_layers"]
            
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, inputs, maybe_arg=2):
        # GET THE TITLE TEXT
        s_slide = '。'.join(inputs.split())
        s_slide = re.sub(r"|[+]|[-]|[;]|[#]|[/]|[0-9]","",s_slide)
        
        # TOKENIZE THE INPUT
        tokens = self.tokenizer.tokenize("[CLS] " + s_slide + " [SEP]")
    
        # OBTAIN THE TOKEN_IDS TENSOR        
        tokens_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
            
        # RETURN THE TENSOR INPUT FOR THE MODEL AND THE TOKENS
        for_processing = {"model_input":{"input_ids": tokens_tensor},
                          "tokens":tokens}
        return for_processing

    def _forward(self, model_inputs):
        # RETRIEVE THE VALUES FROM THE PREPROCESS METHOD
        model_input = model_inputs["model_input"]
        
        # PROCESS THE TENSOR INPUT WITH THE BERT MODEL
        output = self.model(**model_input)
        
        # RETURN THE MODEL OUTPUTS AND THE TOKENS
        for_postprocessing = {"output":output,
                              "tokens":model_inputs["tokens"]}
        return for_postprocessing

    def postprocess(self, model_outputs, n_layers=12):
        # RETRIEVE THE VALUES FROM THE FORWARD METHOD
        output = model_outputs["output"]
        tokens = model_outputs["tokens"]
        
        # ESTIMATE A SINGLE EMBEDDING PER TOKEN
        embeddings = output[2]
        embeddings = torch.stack(embeddings, dim=0)
        embeddings = torch.squeeze(embeddings, dim=1)
        embeddings = torch.cat((list(embeddings[-n_layers:])),1)
        
        # CORRECT THE TOKENS (WORD-WISE)
        n_embeddings, n_tokens = self.correct_embeddings(embeddings, tokens)
        
        # DROP [CLS] AND [SEP] TOKENS
        n_tokens = n_tokens[1:-1]
        n_embeddings = n_embeddings[1:-1]
        
        # CORRECT THE TITLE EMBEDDINGS
        nt_tokens, nt_embeddings = self.extract_title(n_tokens,n_embeddings)
        
        # PREPARE CANDIDATES DATAFRAME
        candidates, E_jt = self.extract_candidates(n_tokens,n_embeddings)
        
        candidates ={"candidate":candidates,"embeddings":E_jt}
        
        # RETURN THE ESTIMATED ATTENTION MAP, EMBEDDINGS AND THE TOKENS
        output_dict = {"candidates_info":candidates,
                       "title_tokens":nt_tokens,
                       "embeddings_title":nt_embeddings}
        return output_dict
    
    def correct_embeddings(self, prev_embeddings, prev_tokens):
        # AUXILIARY VARIABLES
        i = 0
        j = 1
        aux_o = 0
        aux_f = 0
        aux_bool = False
        test_list = []
        n_repeats = 0
        new_tokens = []
        new_indices = []
        indices_len = []
        
        # MAIN LOOP FOR CORRECTING TOKENS
        for token in prev_tokens:
            # WHETHER THE TOKEN STARTS WITH AN "##"
            if not(token == token.replace("##", "")):
                j += 1
                n_repeats += 1
                
                # WHETHER IS A NEW WORD
                if(aux_bool):
                    test_list.append(i)
                    indices_len.pop(-1)
                    indices_len.append(j)
                else:
                    i -= 1
                    test_list.append(i)
                    indices_len.append(j)
                    new_indices.append(i)
                
                # UPDATE THE STATE
                aux_t = new_tokens[-1]
                new_tokens.pop(-1)
                new_tokens.append(aux_t + token.replace("##", ""))
                aux_bool = True
            else:
                if(aux_bool):
                    i += 1
                aux_bool = False
                test_list.append(i)
                new_tokens.append(token)
                j = 1
                i += 1
        if len(new_indices)>=1:           
            # CORRECT THE EMBEDDINGS
            index = torch.tensor(test_list).repeat(prev_embeddings.shape[1],1).T
            new_embeddings = torch.zeros(prev_embeddings.shape[0]-n_repeats, prev_embeddings.shape[1], dtype=prev_embeddings.dtype).scatter_(0,index,prev_embeddings, reduce='add')
            for i_index in range(len(new_indices)):
                n_index = new_indices[i_index]
                n_len = indices_len[i_index]
                new_embeddings[n_index]/=n_len
        else:
            # CORRECTION IS NOT REQUIRED
            new_embeddings = prev_embeddings
            new_tokens = prev_tokens
        
        # RETURN THE CORRECTED VALUES
        return new_embeddings,new_tokens
    
    def extract_candidates(self,slide_words,words_embeddings):
        
        # AUXILIARY VARIABLES
        phrases = []
        embeddings = []
        n_words = len(slide_words)
        
        # LOOP THROUGH ALL WORDS
        for i in range(n_words-1):
            mini_doc_1 = nagisa.tagging(slide_words[i])
            mini_doc_2 = nagisa.tagging(slide_words[i+1])
            len_cond_1 = (len(mini_doc_1.postags)<3) and (len(slide_words[i])<10)
            len_cond_2 = (len(mini_doc_2.postags)<3) and (len(slide_words[i+1])<10)
            if ("名詞" in mini_doc_1.postags)and(len_cond_1):
                phrases.append(slide_words[i])
                embeddings.append(torch.unsqueeze(words_embeddings[i],0))
                if ("名詞" in mini_doc_2.postags)and(len_cond_2):
                    phrases.append(slide_words[i]+
                                   slide_words[i+1])
                    auxiliar_embedd = torch.unsqueeze((words_embeddings[i] + words_embeddings[i+1])/2,1)
                    embeddings.append(auxiliar_embedd.T)
        mini_doc_1 = nagisa.tagging(slide_words[n_words-1])
        len_cond_1 = (len(mini_doc_1.postags)<3) and (len(slide_words[n_words-1])<10)
        if ("名詞" in mini_doc_1.postags)and(len_cond_1):
            phrases.append(slide_words[n_words-1])
            embeddings.append(torch.unsqueeze(words_embeddings[n_words-1],0))
        embeddings_t = torch.cat(embeddings,axis=0)
        return phrases,embeddings_t
    
    def extract_title(self,title_tokens,title_embeddings):
        joshi = ["の","と","を","は","が","に","で",
                 "。",".","、",",",":","I","%","短縮", "版",
                 '年度', '春', '学期', '島田', 'A', '課程', '月曜', '限', '目',
                 "Contact","atsushi","ait","kyushuu","ac","IT","社会","ため","論"]
        
        # AUXILIARY VARIABLES
        title_words = []
        embeddings = []
        n_words = len(title_tokens)
        
        # LOOP THROUGH ALL WORDS
        for i in range(n_words):
            mini_doc = nagisa.tagging(title_tokens[i])
            if (("名詞" in mini_doc.postags)and not(title_tokens[i] in joshi)):
                title_words.append(title_tokens[i])
                embeddings.append(torch.unsqueeze(title_embeddings[i],0))
        embeddings_t = torch.cat(embeddings,axis=0)
        return title_words,embeddings_t
    
class EmbeddPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        # ENABLE THE USE OF ARGUMENT TITLE_EMBEDD FOR POSTPROCESSING
        postprocess_kwargs = {}
        if "params" in kwargs:
            postprocess_kwargs["params"] = kwargs["params"]
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, inputs, maybe_arg=2):
        tokens = self.tokenizer.tokenize("[CLS] " + inputs + " [SEP]")
        tokens_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        return {"model_input":{"input_ids":tokens_tensor}}

    def _forward(self, model_inputs):
        model_input = model_inputs["model_input"]
        output = self.model(**model_input)
        for_postprocessing = {"output":output}
        return for_postprocessing

    def postprocess(self, model_outputs,params={"n_layers":12}):
        n_layers = params["n_layers"]
        output = model_outputs["output"]
        embeddings = output[2]
        embeddings = torch.stack(embeddings, dim=0)
        embeddings = torch.squeeze(embeddings, dim=1)
        embeddings = torch.cat((list(embeddings[-n_layers:])),1)
        n_embeddings = embeddings[1:-1]
        return n_embeddings

title_pipe = LECTOR_TITLE_Pipeline(model=model,tokenizer=tokenizer)
slide_pipe = LECTOR_SLIDE_Pipeline(model=model,tokenizer=tokenizer)
text_pipe = EmbeddPipeline(model=model,tokenizer=tokenizer)

def read_slide(course_ID,material,page):
    with open('./EDU_DATA/Lecture Materials/GPT_VER/Course_{}_LectureMaterialText_{}_p_{}.txt'.format(course_ID,material,page), 'r',encoding="utf8") as f:
        textInfo = f.read()
    slide = textInfo
    slide = slide.replace(" ", "")
    return slide

class LECTOR:
    def __init__(self,course_ID,hyperparameters):
        self.course_ID = course_ID
        self.n_layers = hyperparameters["n_layers"]
        self.psi_t = hyperparameters["psi_t"]
        self.psi_b = hyperparameters["psi_b"]
        self.alpha = hyperparameters["alpha"]
        self.beta = hyperparameters["beta"]
        self.d_lineal = hyperparameters["d_lineal"]
        self.content_embedd = text_pipe("内容",params={"n_layers":self.n_layers})
        self.materials_info = pd.read_csv("./EDU_DATA/Students/Course_{}_LectureMaterial.csv".format(course_ID))[["contentsid","pages"]]
        self.LECTOR_material = []
        for index, row in self.materials_info.iterrows():
            self.LECTOR_material.append(LECTOR_material(course_ID,row["contentsid"],row["pages"],self.n_layers,self.psi_t,self.psi_b,self.alpha))
    def process(self):
        raw_candidates = []
        for material in self.LECTOR_material:
            print("Processing material: {}".format(material.material_ID))
            material.process()
#             print(len(list(dict.fromkeys(material.candidates))))
            raw_candidates.extend(material.candidates)
        
        print("Post-processing...")
        # RETRIEVE ALL CANDIDATES
        self.candidates = list(dict.fromkeys(raw_candidates))
        
        # AUXILIAR DATAFRAME TO EXPAND CURRENT DATAFRAMES
        auxDf = pd.DataFrame({"candidate":self.candidates,"aux":[0]*len(self.candidates)})
        
        # FINAL OUTPUTS OF THE MODEL (DATAFRAME VERSION)
        self.cross_title = pd.DataFrame(columns=self.candidates)
        self.cross_body = pd.DataFrame(columns=self.candidates)
        self.self_scores = pd.DataFrame(columns=self.candidates)
        
        # EXPAND THE CURRENT DATAFRAMES TO CONSIDER ALL THE CANDIDATES
        flag_first = True
        for material in self.LECTOR_material:
            if flag_first:
                print("flag_entered")
                self.title_tensor = material.title_tensor
                self.body_tensor = material.body_tensor
                self.candidates_embedd_tensor = material.candidates_embedd_tensor
                
                self.df_indexes = material.df_indexes
                self.non_filtered_candidates = material.candidates[:]
                flag_first = False
            else:
                self.title_tensor = torch.cat([self.title_tensor,material.title_tensor],axis=0)
                self.body_tensor = torch.cat([self.body_tensor,material.body_tensor],axis=0)
                self.candidates_embedd_tensor = torch.cat([self.candidates_embedd_tensor,material.candidates_embedd_tensor],axis=0)
                
                self.df_indexes = pd.concat([self.df_indexes,material.df_indexes])
                self.non_filtered_candidates.extend(material.candidates)
            
            self_scores = material.self_scores.merge(auxDf, how='right', on="candidate").fillna(0)
            self_scores = self_scores.drop(columns=["aux"])
            self_scores = self_scores.set_index("candidate")
            self_scores = self_scores.transpose()
            self_scores = self_scores.reset_index()
            self_scores["material"] = material.material_ID
            self.self_scores = pd.concat([self.self_scores,self_scores],axis=0)

        self.df_indexes = self.df_indexes.reset_index(drop=True)
        #  ESTIMATION OF THE CROSS-SCORES
        A_title = pd.DataFrame(self.normalized_attention(self.title_tensor,self.candidates_embedd_tensor).T.numpy())
        A_body = pd.DataFrame(self.normalized_attention(self.body_tensor,self.candidates_embedd_tensor).T.numpy())
        A_title["candidate"] = self.non_filtered_candidates
        A_body["candidate"] = self.non_filtered_candidates
        A_title["aux"] = 1
        A_body["aux"] = 1
        self.cross_title = A_title.set_index(["candidate"]).sum(level=["candidate"]).reset_index()
        self.cross_body = A_body.set_index(["candidate"]).sum(level=["candidate"]).reset_index()

        for i in self.cross_title.drop(columns=["candidate","aux"]).columns:
            self.cross_title[i] = self.cross_title[i]/self.cross_title["aux"]**self.beta ###alpha value here
            self.cross_body[i] = self.cross_body[i]/self.cross_body["aux"]**self.beta ###alpha value here
        
        self.cross_title = self.cross_title.drop(columns=["aux"])
        self.cross_title = self.cross_title.set_index("candidate")
        self.cross_title = self.cross_title.transpose()
        self.cross_title = pd.concat([self.df_indexes,self.cross_title],axis=1)
        
        self.cross_body = self.cross_body.drop(columns=["aux"])
        self.cross_body = self.cross_body.set_index("candidate")
        self.cross_body = self.cross_body.transpose()
        self.cross_body = pd.concat([self.df_indexes,self.cross_body],axis=1)
        
        # MAKE ALL COLUMNS THE SAME
        columns = self.cross_title.columns
        self.cross_body = self.cross_body[columns]
        self.self_scores = self.self_scores[columns]
        
        #NORMALIZE AND CALCULATE FINAL SCORES
        a = self.cross_body.drop(columns=["index","material"]).to_numpy()
        b = self.cross_title.drop(columns=["index","material"]).to_numpy()
        c = a + b
        d = scaler_cross.fit_transform(c.reshape(c.shape[0]*c.shape[1],1))
        self.scaler = scaler_cross
        self.cross_scores = pd.concat([
            self.cross_body[["material","index"]].reset_index(drop=True),
            pd.DataFrame(d.reshape(c.shape[0],c.shape[1]),
                         columns=self.cross_body.drop(columns=["index","material"]).columns)
            ],axis=1)
        e = self.self_scores.drop(columns=["index","material"]).to_numpy()
        f = scaler_self.fit_transform(e.reshape(e.shape[0]*e.shape[1],1))
        self.selfs_scores = pd.concat([
            self.self_scores[["material","index"]].reset_index(drop=True),
            pd.DataFrame(f.reshape(e.shape[0],e.shape[1]),
                         columns=self.self_scores.drop(columns=["index","material"]).columns)
            ],axis=1)
        
        r = (1-self.d_lineal)*d + self.d_lineal*f ## d value
        
        self.final_scores = pd.concat([
            self.cross_body[["material","index"]].reset_index(drop=True),
            pd.DataFrame(r.reshape(c.shape[0],c.shape[1]),
                         columns=self.cross_body.drop(columns=["index","material"]).columns)
            ],axis=1)
        # APPLY THE MMR NORMALIZATION TO ALL CANDIDATES
        self.filt_cand_embedd_tensor, self.filt_cand_list = self.average_embeddings()
        sims = self.normalized_attention( self.filt_cand_embedd_tensor, self.filt_cand_embedd_tensor)
        self.sims_Df = pd.DataFrame(sims.numpy(),columns=self.filt_cand_list,index=self.filt_cand_list)
        
        self.new_final_s = self.final_scores
        
        for index, row in self.materials_info.iterrows():
            material = row["contentsid"]
            n_pages = row["pages"]
            print("Post-processing: {}".format(material))
            for i in tqdm(range(1,n_pages + 1)):
                self.new_final_s = self.modify_bests_by_slide(self.new_final_s,material,i,0.7)
        g = self.new_final_s.drop(columns=["index","material"]).to_numpy()
        h = scaler_final.fit_transform(g.reshape(g.shape[0]*g.shape[1],1))
        self.new_final_s = pd.concat([
            self.new_final_s[["material","index"]].reset_index(drop=True),
            pd.DataFrame(h.reshape(g.shape[0],g.shape[1]),
                         columns=self.new_final_s.drop(columns=["index","material"]).columns)
            ],axis=1)
        
        print("Finished")
    def normalized_attention(self,E_A,E_B):
        norm_A = torch.unsqueeze(torch.linalg.norm(E_A,dim=1,ord=2),1)
        norm_B = torch.unsqueeze(torch.linalg.norm(E_B,dim=1,ord=2),1)
        S = torch.matmul(E_A,E_B.T)/(torch.matmul(norm_A,norm_B.T)+0.0001)
        return S
    def find_duplicates(self):
        lst = self.non_filtered_candidates
        duplicates = defaultdict(list)
        for i, item in enumerate(lst):
            duplicates[item].append(i)
        duplicates_dict = {key: value for key, value in duplicates.items()}
        return duplicates_dict
    def average_embeddings(self):
        duplicates = self.find_duplicates()
        embeddings = self.candidates_embedd_tensor
        averaged_embeddings = []
        original_elements = []
#         list_of_freqs = []
        for key, indices in duplicates.items():
            if len(indices) > 1:
                vectors = embeddings[indices]
                averaged_vector = torch.mean(vectors, dim=0)
                averaged_embeddings.append(averaged_vector)
            else:
                index = indices[0]
                averaged_embeddings.append(embeddings[index])
            original_elements.append(key)
#             list_of_freqs.append(len(indices))
        averaged_tensor = torch.stack(averaged_embeddings)
#         self.freqs_np = pd.DataFrame({'freqs': list_of_freqs}, index=original_elements)
        return averaged_tensor, original_elements
    def modify_bests_by_slide(self,scores_df,material,page,lambda_p):
        result = scores_df.loc[
            (scores_df["material"]==material)&(scores_df["index"]==page)
        ].drop(columns=["material","index"]).reset_index(drop=True).T.sort_values(by=0,ascending=False)
        values = result[0].to_numpy()
        r_list = result.index.tolist()
        
#         cand_freqs = self.freqs_np.loc[r_list, ['freqs']].to_numpy().reshape(values.shape)
        cand_sims = self.sims_Df.loc[r_list,r_list].to_numpy()
        cand_sims[np.triu_indices_from(cand_sims)] = 0
        
        mmr_scores = np.amax(cand_sims, axis=1)
        mmr_scores[0] = mmr_scores.mean()
        mmr_scores = self.scaler.transform(mmr_scores.reshape(mmr_scores.shape[0],1)).reshape(values.shape)
        
        new_values = lambda_p*values - (1-lambda_p)*mmr_scores
#         new_values = new_values * (cand_freqs ** (1-self.beta))
  
        # Create a copy of the scores_df DataFrame
        modified_scores_df = scores_df.copy()

        # Set the new values in the copied DataFrame using loc
        modified_scores_df.set_index(["material", "index"], inplace=True)
        modified_scores_df.loc[(material, page), r_list] = new_values
        modified_scores_df = modified_scores_df.reset_index()

        return modified_scores_df
    def update_mmr_state(self,lambda_p):
        # APPLY THE MMR NORMALIZATION TO ALL CANDIDATES
        self.filt_cand_embedd_tensor, self.filt_cand_list = self.average_embeddings()
        sims = self.normalized_attention( self.filt_cand_embedd_tensor, self.filt_cand_embedd_tensor)
        self.sims_Df = pd.DataFrame(sims.numpy(),columns=self.filt_cand_list,index=self.filt_cand_list)
        
        self.new_final_s = self.final_scores
        
        for index, row in self.materials_info.iterrows():
            material = row["contentsid"]
            n_pages = row["pages"]
            print("Post-processing: {}".format(material))
            for i in tqdm(range(1,n_pages + 1)):
                self.new_final_s = self.modify_bests_by_slide(self.new_final_s,material,i,lambda_p)
        g = self.new_final_s.drop(columns=["index","material"]).to_numpy()
        h = scaler_final.fit_transform(g.reshape(g.shape[0]*g.shape[1],1))
        self.new_final_s = pd.concat([
            self.new_final_s[["material","index"]].reset_index(drop=True),
            pd.DataFrame(h.reshape(g.shape[0],g.shape[1]),
                         columns=self.new_final_s.drop(columns=["index","material"]).columns)
            ],axis=1)

        print("Finished")
    def extract_topics(self,n_ex_topics,mmr_alpha):
        for material in self.LECTOR_material:
            print("Processing material: {}".format(material.material_ID))
            material.extract_topics_without_filter(self.final_scores,self.self_scores,n_ex_topics)  #extract_topics(self.final_scores,self.self_scores,n_ex_topics,mmr_alpha)

class LECTOR_material:
    def __init__(self,course_ID, material_ID, material_pages,n_layers,psi_t,psi_b,alpha):
        self.course_ID = course_ID
        self.material_ID = material_ID
        self.material_pages = material_pages
        self.n_layers = n_layers #
        self.psi_t = psi_t #
        self.psi_b = psi_b
        self.alpha = alpha #
        self.title_tensor = torch.zeros(1,n_layers*1024)
        self.body_tensor = torch.zeros(1,n_layers*1024)
        self.attention_Df = []
        self.slide_wise_candidates = []
    def process(self):
        page = 1
        slide = read_slide(self.course_ID,self.material_ID,page)
        self.title_info = title_pipe(slide,n_layers=self.n_layers)
        self.candidates = self.title_info["candidates_info"]["candidate"][:]
        self.slide_wise_candidates.append(self.title_info["candidates_info"]["candidate"][:])
        self.candidates_embedd_tensor = self.title_info["candidates_info"]["embeddings"][:]
        for page in tqdm(range(2,self.material_pages+1)):
            slide_2 = read_slide(self.course_ID,self.material_ID,page)
            params = {"title_embedd":self.title_info["embeddings_title"],"n_layers":self.n_layers,"psi_t":self.psi_t,"psi_b":self.psi_b}
            result = slide_pipe(slide_2,params=params)
            self.update_state(result)
        print("Finishing...")
        
        #  ESTIMATION OF THE CROSS-SCORES
        self.df_indexes = pd.DataFrame({"material":[self.material_ID]*self.material_pages,"index":list(range(1,self.material_pages+1))})
        
        # ESTIMATION OF THE SELF-SCORES
        candidates_filtered = list(dict.fromkeys(self.candidates))
        self.self_scores = pd.DataFrame({"candidate":candidates_filtered,0:[0]*len(candidates_filtered)})
        for i in range(1,self.material_pages):
            self.self_scores = self.attention_Df[i-1].merge(self.self_scores, how='right', on="candidate").fillna(0)
            self.self_scores[i] = self.self_scores["score"]
            self.self_scores = self.self_scores.drop(columns=["count","attention","score"])
    def update_state(self,result):
        candidates_info = result["candidates_info"]
        self.candidates_embedd_tensor = torch.cat([self.candidates_embedd_tensor,candidates_info["embeddings"]],axis=0)
        self.candidates.extend(candidates_info["candidate"])
        self.slide_wise_candidates.append(candidates_info["candidate"])
        self.title_tensor = torch.cat([self.title_tensor,result["embeddings_title"]],axis=0)
        self.body_tensor = torch.cat([self.body_tensor,result["embeddings_body"]],axis=0)
        candidates_table = pd.DataFrame({"candidate":candidates_info["candidate"], "attention":candidates_info["importance"]})
        cand_table = candidates_table["candidate"].value_counts().to_frame()
        cand_table_2 = candidates_table.set_index(['candidate'])["attention"].sum(level="candidate").to_frame()
        cand_table = cand_table.merge(cand_table_2, how='inner', left_index=True,right_index=True)
        cand_table = cand_table.reset_index()
        cand_table.columns = ["candidate","count","attention"]
        cand_table["score"] = cand_table["attention"]/(cand_table["count"]+self.alpha)
        self.attention_Df.append(cand_table)
    def obtain_bests_by_slide(self,scores_df,material,page,n_best):
        result = scores_df.loc[(scores_df["material"]==material)&(scores_df["index"]==page)].drop(columns=["material","index"]).reset_index(drop=True).T.sort_values(by=0,ascending=False)[0:n_best]
        r_list = result.index.tolist()
        return r_list
    def normalized_attention(self,E_A,E_B):
        norm_A = torch.unsqueeze(torch.linalg.norm(E_A,dim=1,ord=2),1)
        norm_B = torch.unsqueeze(torch.linalg.norm(E_B,dim=1,ord=2),1)
        S = torch.matmul(E_A,E_B.T)/(torch.matmul(norm_A,norm_B.T))
        return S
    def cosine_similarity(self,matrix):
        dot_product = np.dot(matrix, matrix.T)
        norm = np.linalg.norm(matrix, axis=1)
        norm_matrix = np.outer(norm, norm)
        similarity_matrix = dot_product / norm_matrix
        return similarity_matrix
    def gaussian_similarity_matrix(self,num_elements,mean,variance):
        # Create an identity matrix with dimensions num_elements x num_elements
        similarity_matrix = np.zeros((num_elements, num_elements))
        np.fill_diagonal(similarity_matrix, mean)

        # Calculate Gaussian decay values for each pair of elements
        for i in range(num_elements):
            for j in range(i + 1, num_elements):
                decay = np.exp(-((i - j- mean) ** 2) / (2 * variance))
                similarity_matrix[i, j] = decay
                similarity_matrix[j, i] = decay  # Similarity matrix is symmetric

        return similarity_matrix
    
    def extract_topics_without_filter(self,f_r_LECTOR,s_r_LECTOR,n_ex_topics):

        f_LECTOR = f_r_LECTOR.copy()
        s_LECTOR = s_r_LECTOR.copy()

        material_LECTOR = f_LECTOR.loc[
            f_LECTOR["material"]==self.material_ID
            ].set_index(["material","index"])[list(dict.fromkeys(self.candidates))]
        material_LECTOR[material_LECTOR<0] = 0
        material_LECTOR = material_LECTOR.reset_index(drop=False)
        
#         print("material LECTOR: {}".format(material_LECTOR))
        
        #SPECIFIC TOPICS: SLIDE-WISE BEST TOPICS
        r_cands = []
        for page_i in range(2,self.material_pages+1):
            r_cands.extend(self.obtain_bests_by_slide(material_LECTOR,self.material_ID,page_i,5))
        s_cands = list(dict.fromkeys(r_cands))
        class_1_t_matrix = material_LECTOR[s_cands].to_numpy()[1:]
        
#         print("class_1_t_matrix: {}".format(class_1_t_matrix))
        
        # ANALYSIS OF LATENT TOPICS
        self.loss_list = []
        co_ocurr_sets = []
        
        # Creation of Latent topics descriptors: mean, std, max
        print("LATENT TOPIC DESCRIPTION")
        latent_max = int(len(s_cands)/4)+2
        for n_components in tqdm(range(2,latent_max)):
            for n_trial in range(10):
                # Initialize and fit the NMF model
                model = NMF(n_components=n_components, init='random', random_state=n_trial, max_iter=5000)
                W = model.fit_transform(class_1_t_matrix)
                H = model.components_

                # Loss logging
                self.loss_list.append(model.reconstruction_err_)

                # IN-TOPIC normalization
                new_H = torch.tensor(H)/torch.unsqueeze(torch.amax(torch.tensor(H),dim=1), 1)
                new_H = new_H.numpy()

                # Describing results
                r_H_df = pd.DataFrame(new_H,columns=s_cands)
                n_results = r_H_df.describe().loc[["mean","std","max"]].reset_index(drop=False)

                #New documents generation
                for i in range(n_components):
                    n_doc = r_H_df.T.sort_values(by=i,ascending=False).reset_index()["index"][0:int(len(s_cands)/n_components)].to_list()
                    co_ocurr_sets.append(n_doc)

                # topic characteristics
                if(n_components == 2):
                    results_df = n_results
                else:
                    results_df = pd.concat([n_results,results_df],axis=0)
        
        #DESCRIPTORS DATAFRAME
        mean_max = results_df.loc[results_df["index"]=="max"].describe().T.sort_values(by="mean",ascending=False)
        scaler = MinMaxScaler()
        mean_max[["mean","std","min"]] = scaler.fit_transform(mean_max[["mean","std","min"]])
        mean_max = mean_max.sort_values(by="mean",ascending=False)
        self.ex_topics = mean_max

    


    def extract_topics(self,f_r_LECTOR,s_r_LECTOR,n_ex_topics,mmr_alpha):

        f_LECTOR = f_r_LECTOR.copy()
        s_LECTOR = s_r_LECTOR.copy()

        material_LECTOR = f_LECTOR.loc[
            f_LECTOR["material"]==self.material_ID
            ].set_index(["material","index"])[list(dict.fromkeys(self.candidates))]
        material_LECTOR[material_LECTOR<0] = 0
        material_LECTOR = material_LECTOR.reset_index(drop=False)
        
#         print("material LECTOR: {}".format(material_LECTOR))
        
        #SPECIFIC TOPICS: SLIDE-WISE BEST TOPICS
        r_cands = []
        for page_i in range(2,self.material_pages+1):
            r_cands.extend(self.obtain_bests_by_slide(material_LECTOR,self.material_ID,page_i,5))
        s_cands = list(dict.fromkeys(r_cands))
        class_1_t_matrix = material_LECTOR[s_cands].to_numpy()[1:]
        
#         print("class_1_t_matrix: {}".format(class_1_t_matrix))
        
        # ANALYSIS OF LATENT TOPICS
        self.loss_list = []
        co_ocurr_sets = []
        
        # Creation of Latent topics descriptors: mean, std, max
        print("LATENT TOPIC DESCRIPTION")
        latent_max = int(len(s_cands)/4)+2
        for n_components in tqdm(range(2,latent_max)):
            # Initialize and fit the NMF model
            model = NMF(n_components=n_components, init='random', random_state=0, max_iter=5000)
            W = model.fit_transform(class_1_t_matrix)
            H = model.components_

            # Loss logging
            self.loss_list.append(model.reconstruction_err_)

            # IN-TOPIC normalization
            new_H = torch.tensor(H)/torch.unsqueeze(torch.amax(torch.tensor(H),dim=1), 1)
            new_H = new_H.numpy()

            # Describing results
            r_H_df = pd.DataFrame(new_H,columns=s_cands)
            n_results = r_H_df.describe().loc[["mean","std","max"]].reset_index(drop=False)

            #New documents generation
            for i in range(n_components):
                n_doc = r_H_df.T.sort_values(by=i,ascending=False).reset_index()["index"][0:int(len(s_cands)/n_components)].to_list()
                co_ocurr_sets.append(n_doc)

            # topic characteristics
            if(n_components == 2):
                results_df = n_results
            else:
                results_df = pd.concat([n_results,results_df],axis=0)
        
        #DESCRIPTORS DATAFRAME
        mean_max = results_df.loc[results_df["index"]=="max"].describe().T.sort_values(by="mean",ascending=False)
        mean_mean = results_df.loc[results_df["index"]=="mean"].describe().T.sort_values(by="mean",ascending=False)
        max_max = results_df.loc[results_df["index"]=="max"].describe().T.sort_values(by="max",ascending=False)
        scaler = MinMaxScaler()
        mean_max[["mean","std","min"]] = scaler.fit_transform(mean_max[["mean","std","min"]])
        mean_mean[["mean","std","min"]] = scaler.fit_transform(mean_mean[["mean","std","min"]])
        max_max[["max","std","min"]] = scaler.fit_transform(max_max[["max","std","min"]])
        mean_max = mean_max.sort_values(by="mean",ascending=False)
        mean_mean = mean_mean.sort_values(by="mean",ascending=False)
        max_max = max_max.sort_values(by="max",ascending=False)
        
        if (self.material_pages>=10):
            #CO-OCURRENCE MATRIX DEFINITION
            #SLIDE-BASED CO-OCURRENCE
            merged_list = [item for sublist in co_ocurr_sets for item in sublist]
            merged_list = list(dict.fromkeys(merged_list))
            slide_co_oc = self.slide_wise_candidates[:]
            f_slide_co_oc = []
            for slide_c_list in slide_co_oc:
                cmn_elements = [element for element in slide_c_list if element in merged_list]
                cmn_elements = list(dict.fromkeys(cmn_elements))
                f_slide_co_oc.append(cmn_elements)
            
            second_slide_list = material_LECTOR.loc[material_LECTOR["index"]==2].drop(columns=["material","index"]).T.reset_index().sort_values(by=1,ascending=False)[0:10]["index"].to_list()
            
            if ("内容" in second_slide_list):
                n_init = 2
            else:
                n_init = 1

            #CLUSTER-BASED CO-OCURRENCE
            s_embedds = self.title_tensor[n_init:] + self.body_tensor[n_init:]
            s_sims = self.normalized_attention(s_embedds,s_embedds)
            s_embedds_2 = torch.tensor(s_LECTOR.loc[s_LECTOR["material"]==self.material_ID][list(dict.fromkeys(self.candidates))].to_numpy()[n_init:])
            s_sims_2 = self.normalized_attention(s_embedds_2,s_embedds_2)


            #UMAP CLUSTERING (BERTopic LIKE PROCESSING)
            n_trials = int(s_embedds.shape[0]*0.6)
            cluster_groups = []
            unwanted_children = []

            print("UMAP CLUSTERIZATION")
            for i in tqdm(range(n_trials)):
                # CLUSTERS FROM THE EMBEDDING REPRESENTATION
                reducer = umap.UMAP(n_components=i+2,random_state=0)
                embedding = reducer.fit_transform(s_embedds)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
                clusterer.fit(embedding)

                # Create a list of lists to hold the indices for each cluster
                clusters_list = [[] for _ in range(np.max(clusterer.labels_) + 1)]

                for idx, label in enumerate(clusterer.labels_):
                    if label != -1:
                        clusters_list[label].append(idx+3)
                    else:
                        unwanted_children.append(idx+3)

                cluster_groups.extend(clusters_list)

                # CLUSTERS FROM THE LECTOR IMPORTANCE REPRESENTATION
                reducer = umap.UMAP(n_components=i+2,random_state=0)
                embedding = reducer.fit_transform(s_embedds_2)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
                clusterer.fit(embedding)

                # Create a list of lists to hold the indices for each cluster
                clusters_list = [[] for _ in range(np.max(clusterer.labels_) + 1)]

                for idx, label in enumerate(clusterer.labels_):
                    if label != -1:
                        clusters_list[label].append(idx+3)
                    else:
                        unwanted_children.append(idx+3)

                cluster_groups.extend(clusters_list)
    #         print("cluster_groups: {}".format(cluster_groups))
            merged_list_slides = [item for sublist in cluster_groups for item in sublist]
            merged_list_slides = list(dict.fromkeys(merged_list_slides))
            merged_list_slides = sorted(merged_list_slides)
            co_oc_slides = np.zeros((len(merged_list_slides),len(merged_list_slides)))

            print("SLIDE CO-OCURRENCE MAP")
            for n in tqdm(range(len(merged_list_slides))):
                n_word_s = merged_list_slides[n]
                for sample in cluster_groups:
                    if(n_word_s in sample):
                        for m_word in sample:
                            index = np.argwhere(np.array(merged_list_slides) == m_word)[0][0]
                            co_oc_slides[n][index] += 1
            co_ocs_slides = pd.DataFrame(co_oc_slides,columns=merged_list_slides,index=merged_list_slides)

            # CONSECUTIVE SLIDES BIAS ADDITION
            # Number of elements
            num_elements = len(merged_list_slides)
            # Mean and variance for Gaussian decay (you can adjust these values)
            mean_value = 1
            variance_value = 3
            # Create the similarity matrix
            co_oc_henshu = self.gaussian_similarity_matrix(num_elements, mean=mean_value, variance=variance_value)
            #Integration
            alpha_slide = 0.25
            self.sim_matrix = (1-alpha_slide)*self.cosine_similarity(co_oc_slides)+alpha_slide*co_oc_henshu
            linkage_matrix = linkage(1-self.sim_matrix,method='average',metric='cosine')

            # HIERARCHICAL CLUSTERIZATION
            f_cluster_co_oc = []
            n_hierar = int(s_embedds.shape[0]*0.6)+2
            print("HIERARCHICAL CLUSTERIZATION")
            for n_clusters in tqdm(range(2,n_hierar)):
                clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
                cluster_labels = clustering.fit_predict(1-self.cosine_similarity(co_oc_slides))
                # Create a list of lists to hold the indices for each cluster
                clusters_results = [[] for _ in range(np.max(cluster_labels) + 1)]

                for idx, label in enumerate(cluster_labels):
                    clusters_results[label].append(idx+3)

                for i in range(n_clusters):
                    clust_LECTOR = material_LECTOR.loc[(material_LECTOR["material"]==self.material_ID)&
                                                    (material_LECTOR["index"].isin(clusters_results[i]))][s_cands]
                    doc_coocs = clust_LECTOR.sum().reset_index(drop=False).sort_values(by=0,ascending=False)[0:int(len(s_cands)/n_clusters)]["index"].tolist()
                    f_cluster_co_oc.append(doc_coocs)
            self.slide_clusters = f_cluster_co_oc[:]
            co_ocurr_sets.extend(f_slide_co_oc[n_init:])
            co_ocurr_sets.extend(f_cluster_co_oc)
        merged_list = [item for sublist in co_ocurr_sets for item in sublist]
        merged_list = list(dict.fromkeys(merged_list))
        co_oc = np.zeros((len(merged_list),len(merged_list)))
        
        print("FINAL CO-OCURRENCE MAP")
        for n in tqdm(range(len(merged_list))):
            n_word = merged_list[n]
            for sample in co_ocurr_sets:
                if(n_word in sample):
                    for m_word in sample:
                        index = np.argwhere(np.array(merged_list) == m_word)[0][0]
                        co_oc[n][index] += 1
        self.co_oc_sim_matrix = self.cosine_similarity(co_oc)
        linkage_matrix_topics = linkage(1-self.cosine_similarity(co_oc),method='average',metric='cosine')
        co_ocs = pd.DataFrame(co_oc,columns=merged_list,index=merged_list)
        co_ocs = pd.DataFrame(scaler.fit_transform(co_ocs),columns=merged_list,index=merged_list)
        
        # EXTRACT "N" TOPICS
        n_topics = min(n_ex_topics,co_oc.shape[0])
        alpha_1 = mmr_alpha
        alpha_2 = mmr_alpha

        self.ex_topics = []

        aux_n_topics = 0

        for i in range(n_topics):
            # MEAN_MAX ELEMENT ADDITION
            mean_max["mmr"] = mean_max["mean"]

            co_ocs["max_mmr"] = co_ocs[self.ex_topics].max(axis=1)
            co_ocs["max_mmr"].fillna(0, inplace=True)

            for index, row in mean_max.iterrows():
                try:
                    row["mmr"] = alpha_1*row["mean"] - (1-alpha_1)*co_ocs["max_mmr"].loc[index]
                except:
                    row["mmr"] = -10

        #     print(mean_max.drop(ex_topics).sort_values(by="mmr",ascending=False)[0:10])
            new_topic = mean_max.drop(self.ex_topics).sort_values(by="mmr",ascending=False)[0:1].index[0]
            self.ex_topics.append(new_topic)
            print(new_topic)

            aux_n_topics += 1

            if (aux_n_topics>=n_topics):
                break

            # MEAN_MEAN ELEMENT ADDITION
            mean_mean["mmr"] = mean_mean["mean"]

            co_ocs["max_mmr"] = co_ocs[self.ex_topics].max(axis=1)
            co_ocs["max_mmr"].fillna(0, inplace=True)

            for index, row in mean_mean.iterrows():
                try:
                    row["mmr"] = alpha_2*row["mean"] - (1-alpha_2)*co_ocs["max_mmr"].loc[index]
                except:
                    row["mmr"] = -10

        #     print(mean_mean.drop(ex_topics).sort_values(by="mmr",ascending=False)[0:10])
            new_topic = mean_mean.drop(self.ex_topics).sort_values(by="mmr",ascending=False)[0:1].index[0]
            self.ex_topics.append(new_topic)
            print(new_topic)

            aux_n_topics += 1

            if (aux_n_topics>=n_topics):
                break
        print("Finished")
    
    def consensus_clustering(self, X, h_trials,cdf_resolution):
        #CONSENSUS CLUSTERING
        n_samples = X.shape[0]
        H = h_trials
        K = int(n_samples*0.8)
        M_lists = []
        # FIND THE CONSENSUS MATRIX CORRESPONDING TO K
        # print("Consensus clustering: START")
        for k in range(2,K+1):
            M = np.zeros((n_samples,n_samples))
            I = np.zeros((n_samples,n_samples))
            for h in range(H):
                #Resampler
                np.random.seed(h)
                min_samples = k + 2
                max_samples = n_samples -2
                num_to_keep = np.random.randint(min_samples, max_samples + 1)
                selected_samples = np.random.choice(n_samples, num_to_keep, replace=False)
                d_X = X[selected_samples.tolist(),:]
                # Normalization term
                I[selected_samples[:, None],selected_samples] += 1
                # Clustering
                # kmeans = KMeans(n_clusters=k, random_state=0).fit(d_X)
                kmeans = AgglomerativeClustering(n_clusters=k).fit(d_X)
                for a_label in np.unique(kmeans.labels_):
                    indices = np.where(kmeans.labels_ == a_label)
                    cluster_indices = selected_samples[indices]
                    # M <- M U M^h
                    M[cluster_indices[:, None],cluster_indices] += 1
            M_lists.append(M/I)
        # CALCULATE THE CONSENSUS MATRIX CDFs
        # print("Consensus clustering: ANALYSIS")
        n_samples = cdf_resolution
        index_r = []
        cdf_r = []
        delta_k = []
        k = []

        for j in range(9):
            index_val = []
            cdf_val = []

            for i in range (n_samples+1):
                c_val = i/n_samples
                index_val.append(c_val)
                cdf_val.append(CDF_meassure(M_lists[j],c_val))
            
            index_r.append(index_val)
            cdf_r.append(cdf_val)
            # delta(K) = {A(K) if K=2, (A(K+1)-A(K))/A(K) if K>2
            # updated to : delta(K) = A(K+1)-A(K), A(K=1)=0
            A_k = np.trapz(y=np.array(cdf_val), x=np.array(index_val))
            if j==0:
                delta_k.append(A_k)
            else:
                delta_k.append(A_k - A_k_prev)
            k.append(A_k)
            A_k_prev = A_k
        # PLOT THE RESULTS
        plt.close()
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.plot(index_r[0],cdf_r[0],lw=2.5)
        plt.plot(index_r[1],cdf_r[1],lw=2.5)
        plt.plot(index_r[2],cdf_r[2],lw=2.5)
        plt.plot(index_r[3],cdf_r[3],lw=2.5)
        plt.plot(index_r[4],cdf_r[4],lw=2.5)
        plt.plot(index_r[5],cdf_r[5],lw=2.5)
        plt.plot(index_r[6],cdf_r[6],lw=2.5)
        plt.plot(index_r[7],cdf_r[7],lw=2.5)
        plt.plot(index_r[8],cdf_r[8],lw=2.5)
        plt.subplot(132)
        plt.plot(np.arange(len(delta_k))+2,delta_k,lw=2.5)
        plt.subplot(133)
        plt.plot(np.arange(len(delta_k))+2,k,lw=2.5,linestyle='dashed')

        # print("delta k: {}".format(delta_k))
        # print("mean: {}".format(np.mean(delta_k[1:])))
        # print("index: {}".format(np.argmax(np.array(delta_k) < np.mean(delta_k[1:]))))

        K_opt = np.argmax(np.array(delta_k) < np.mean(delta_k[1:]))

        # STABILITY SPEED INDICATOR
        score = np.trapz(y=np.array(k), x=np.arange(len(delta_k)))/(len(delta_k)-1)

        #RETURN THE M MATRICES FOR FURTHER ANALYSIS
        return M_lists, score, K_opt