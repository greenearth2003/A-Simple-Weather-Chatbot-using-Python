from transformers import BertTokenizer, BertForTokenClassification, BertModel
import torch
from torch.nn.functional import softmax
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from transformers import pipeline
import os



tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
model_ner = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
bert_model = BertModel.from_pretrained('bert-base-uncased')
NER_processing = pipeline("ner", model=model_ner, tokenizer=tokenizer)

# Function to extract name using BERT NER
def extract_name(text):
    ner_results = NER_processing(text)
    
    # Select tokens which have entity tag is "I-PER" (Personal name)
    name_tokens = [entity['word'] for entity in ner_results if entity['entity']=="I-PER"]
    name_eles = []
    
    # Because some Vietnamese names can be tokenized into subwords, they need to be reprocessed.
    for i in range(len(name_tokens)):
        if name_tokens[i][:2] != "##":            
            name_eles.append(name_tokens[i])
        else:
            name_eles[-1] = name_eles[-1] + name_tokens[i][2:]
            
    name = " ".join(name_ele for name_ele in name_eles)
    return name if name else None


def get_embedding(sentence):
    tokens = tokenizer.tokenize(sentence)
    # Add [CLS] and [SEP] token into original sentence
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    T=15
    
    # Sentence padding with max length of sentence is 15 tokens (includes [CLS] and [SEP])
    padded_tokens = tokens + ['[PAD]' for _ in range(T-len(tokens))]
    
    # Create attention mask for real tokens (no padding tokens)
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]

    seg_ids = [0 for _ in range(len(padded_tokens))]

    sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0)

    output = bert_model(token_ids, attention_mask=attn_mask,token_type_ids=seg_ids)
    last_hidden_state, pooler_output = output[0], output[1]
    
    # Only use the embedding vector of [CLS] token as feature
    return pooler_output.detach().numpy()


# Function to extract embedding vectors fors predefined input sentences 
def get_predefined_input_embeddings(file_path, emb_file_path):
    if not os.path.exists(emb_file_path):
        predef_inputs_emb = []
        with open(file_path, 'r', encoding='utf-8') as file: 
            for line in file:
                line_emb = get_embedding(line)
                predef_inputs_emb.append(line_emb)
        np.save(emb_file_path, np.array(predef_inputs_emb))
    
# Function to find similar question for user input
def find_similar_question(user_input, emb_file_path):
    try: 
        predef_input_embs = np.load(emb_file_path)
        user_embedding = get_embedding(user_input)

        similarities = [cosine_similarity(user_embedding, predef_input_embs[i])[0][0] for i in range(len(predef_input_embs))]
        best_match_idx = np.argmax(similarities)
        print(similarities)
        return best_match_idx
    except:
        print("Error: Not exist embedding file of predefined inputs!!!")
        return 0
        
    