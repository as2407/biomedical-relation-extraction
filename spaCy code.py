import re
import json
import csv
import spacy
from spacy.scorer import Scorer
from spacy.training import Example
from spacy.tokens import DocBin

nlp = spacy.load("en_core_web_sm") 

def ner_data_prep(data_file):
    data_initial=[]
    g_pos=[]
    d_pos=[]
    with open(data_file,'r') as f:
        for line in f:
            data={}
            d=json.loads(line)
            pos = []
            data['text'] = d['text']
            g_pos = (d['h']['pos'][0],d['h']['pos'][0] + d['h']['pos'][1],'GENE')
            d_pos = (d['t']['pos'][0],d['t']['pos'][0] + d['t']['pos'][1],'DISEASE')
            pos.append(g_pos)
            pos.append(d_pos)
            data['entities'] = pos
            data_initial.append(data)
    return data_initial





train_data_initial = ner_data_prep("TBGA_train.txt")
dev_data_initial = ner_data_prep("TBGA_val.txt")
test_data_initial = ner_data_prep("TBGA_test.txt")





print(len(train_data_initial),len(dev_data_initial),len(test_data_initial))





spacy_train_ner = [(data['text'], {'entities': data['entities']}) for data in train_data_initial]





with open('ner_train.json', 'w') as f:
    json.dump(spacy_train_ner,f)





spacy_dev_ner = [(data['text'], {'entities': data['entities']}) for data in dev_data_initial]





with open('ner_dev.json', 'w') as f:
    json.dump(spacy_dev_ner,f)





spacy_test_ner = [(data['text'], {'entities': data['entities']}) for data in test_data_initial]





with open('ner_test.json', 'w') as f:
    json.dump(spacy_test_ner,f)





get_ipython().system('python convert.py en ner_train.json ner_train.spacy')
get_ipython().system('python convert.py en ner_dev.json ner_dev.spacy')
get_ipython().system('python convert.py en ner_test.json ner_test.spacy')





get_ipython().run_cell_magic('time', '', '!python -m spacy train config.cfg --output ./ --paths.train ner_train.spacy --paths.dev ner_dev.spacy --training.eval_frequency 1000 --training.max_steps 100000\n')





def evaluate(samples):
    ner_model = spacy.load("ner_model/model-best")
    scorer = Scorer(ner_model)
    example = []
    ner_model = spacy.load("ner_model/model-best")
    for sample in samples:
        pred = ner_model(sample['text'])
        temp_ex = Example.from_dict(pred, {'entities': sample['entities']})
        example.append(temp_ex)
    scores = scorer.score(example)
    
    return scores





ner_test_results = evaluate(test_data_initial)
ner_test_results


# # Relation




def get_word_indices(document, word):
    pattern = r'\b{}\b'.format(re.escape(word))
    matches = re.finditer(pattern, document)
    start_index = -1
    end_index = -1
    for match in matches:
        start_index = len(document[:match.start()].split())
        end_index = len(document[:match.end()].split()) - 1
        break
    if start_index == -1:
        return -1, -1
    else:
        return start_index, end_index





def re_data_prep(data_file):
    relations_data = []
    g_pos = []
    d_pos = []
    
    with open(data_file,'r') as f:
        for line in f:
            data={}
            d=json.loads(line)
            pos = []

            data['document'] = d['text']

            gene_tok={}
            g_pos = (d['h']['pos'][0],d['h']['pos'][0] + d['h']['pos'][1],'GENE')
            gene_tok["text"] = d["text"][g_pos[0]:g_pos[1]]
            gene_tok["start"] = g_pos[0]
            gene_tok["end"] = g_pos[1]

            get_indices = lambda d, t: [i for i, w in enumerate(d.split()) if w in t]

            gene_start = get_word_indices(d['text'], gene_tok["text"])[0]
            gene_end = get_word_indices(d['text'], gene_tok["text"])[1]

            gene_tok["token_start"] = gene_start
            gene_tok["token_end"] = gene_end
            gene_tok["entityLabel"] = "GENE"

            if gene_start!=-1 and gene_end!=-1:
                pos.append(gene_tok)

            disease_tok={}    
            d_pos = (d['t']['pos'][0],d['t']['pos'][0] + d['t']['pos'][1],'DISEASE')
            disease_tok["text"] = d["text"][d_pos[0]:d_pos[1]]
            disease_tok["start"] = d_pos[0]
            disease_tok["end"] = d_pos[1]

            get_indices = lambda d, t: [i for i, w in enumerate(d.split()) if w in t]

            disease_start = get_word_indices(d['text'], disease_tok["text"])[0]
            disease_end = get_word_indices(d['text'], disease_tok["text"])[1]

            disease_tok["token_start"] = disease_start
            disease_tok["token_end"] = disease_end
            disease_tok["entityLabel"] = "DISEASE"

            if disease_start!=-1 and disease_end!=-1:
                pos.append(disease_tok)
            
            data['tokens'] = pos

            if(len(pos)==2):
                relation = {}
                relation['child'] = pos[1]['token_start']
                relation['head'] = pos[0]['token_start']
                relation['relationLabel'] = d['relation']
                data['relations'] = [relation]
                relations_data.append(data)
                
    return relations_data





get_ipython().run_line_magic('cd', 'NLPProject')





train_data_rel = re_data_prep("TBGA_train.txt")
dev_data_rel = re_data_prep("TBGA_val.txt")
test_data_rel = re_data_prep("TBGA_test.txt")





with open('train_relations.json', 'w') as f:
    json.dump(train_data_rel,f)
with open('dev_relations.json', 'w') as f:
    json.dump(dev_data_rel,f)
with open('test_relations.json', 'w') as f:
    json.dump(test_data_rel,f)





with open('train_relations.txt', 'w') as f:
    json.dump(train_data_rel,f)
with open('dev_relations.txt', 'w') as f:
    json.dump(dev_data_rel,f)
with open('test_relations.txt', 'w') as f:
    json.dump(test_data_rel,f)





get_ipython().system('python train_binary_converter.py relations_train.txt relations_train.spacy')
get_ipython().system('python dev_binary_converter.py relations_dev.txt relations_dev.spacy')
get_ipython().system('python test_binary_converter.py relations_test.txt relations_test.spacy')





get_ipython().system('spacy project run train_cpu')





get_ipython().system('spacy project run evaluate')





colors = {"GENE": "#7D8BF6", "DISEASE": "#7DF6D9"}
options = {"colors": colors} 





def NER_Predictions(sentence):
    nlp_ner = spacy.load("ner_model/model-best")
    doc = nlp_ner(sentence)
    return doc





def RE_Predictions(doc):
    # Load the REL model
    nlp2 = spacy.load("training3/model-best")
    nlp2.add_pipe('sentencizer')

    # Run the REL model on the document
    for name, proc in nlp2.pipeline:
        doc = proc(doc)

    # Find all pairs of gene and disease entities and run REL on each pair
    for sent in doc.sents:
        entities = [ent for ent in sent.ents if ent.label_ in ['GENE', 'DISEASE']]
        genes = [ent for ent in entities if ent.label_ == 'GENE']
        diseases = [ent for ent in entities if ent.label_ == 'DISEASE']
        for gene in genes:
            for disease in diseases:
                rel_dict = doc._.rel.get((gene.start, disease.start))
                if rel_dict:
                    predicted_relation = max(rel_dict, key=rel_dict.get)
                    if gene.start > disease.start:
                        gene, disease = disease, gene
                    print(f"Entities: {gene.text, disease.text} =====> Predicted Relation: \033[1;31m{predicted_relation}\033[0m")


user_sentence = input("Enter a Biomedical Context Sentence: \n\n")

# Entities Predictions (NER)
ner_doc = NER_Predictions(user_sentence)
spacy.displacy.render(ner_doc, style="ent", options= options, jupyter=True)

# Relation Predictions (RE)
relation_prediction = RE_Predictions(ner_doc)

ner_model="ner_model/model-best"

def evaluate_ner(ner_model):
    """
    Calculating confusion matrix of Spacy 3.2.1 NER 
    """
    tp,fp,fn,tn = 0,0,0,0

    ner = spacy.load("ner_model/model-best")
    
    with open('ner_test.json', 'r') as f:
        evaluation_data = json.load(f)
    
    for x in evaluation_data:
        # correct ents
        correct_ents = x[1]['entities']
        
        # predicted ents
        doc = ner(x[0])
        predicted_ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
            
        for ent in predicted_ents:
            if ent not in correct_ents:
                pass
        for ent in correct_ents:
            if ent not in predicted_ents:
                pass
                
        # false positives
        fp += len([ent for ent in predicted_ents if ent not in correct_ents])
        # true positives
        tp += len([ent for ent in predicted_ents if ent in correct_ents])
        # false negatives
        fn += len([ent for ent in correct_ents if ent not in predicted_ents])
        # true negatives
        tn += len([ent for ent in correct_ents if ent in predicted_ents])


    test_binary_path = "ner_test.spacy"
    model_path = "ner_model/model-best"
    metrics_path = "ner_metrics.json"

    result = subprocess.run(["python", "-m", "spacy", "evaluate", model_path, test_binary_path, "--output", metrics_path])
    print(result.returncode)

    out = ""
    with open(metrics_path, "r") as fr:
        out = json.load(fr)
        
        for k,v in out.items():
            print(k,v)

    out["True_P"] = tp
    out["False_P"] = fp
    out["False_N"] = fn
    out["True_N"] = tn
    
    return out

evaluate_ner(ner_model)

