import os
import stanfordnlp
from PyPDF2 import PdfReader
from flask import Flask, render_template, request,jsonify
from torch import argmax
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics.text.rouge import ROUGEScore

####### code here to load the trained model from hugging-face hub ########
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
token='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'
tokenizer = AutoTokenizer.from_pretrained("choprahetarth/SemEval-2021-scibert-contributions", use_auth_token=token)
model = AutoModelForSequenceClassification.from_pretrained("choprahetarth/SemEval-2021-scibert-contributions", use_auth_token=token)
##########################################################################

ground_truth = """Automatically validating a research artefact is one of the frontiers in Artificial Intelligence
(AI) that directly brings it close to competing
with human intellect and intuition. Although
criticized sometimes, the existing peer review
system still stands as the benchmark of re-
search validation. The present-day peer review
process is not straightforward and demands
profound domain knowledge, expertise, and
intelligence of human reviewer(s), which is
somewhat elusive with the current state of AI.
However, the peer review texts, which contains
rich sentiment information of the reviewer, re-
flecting his/her overall attitude towards the re-
search in the paper, could be a valuable en-
tity to predict the acceptance or rejection of
the manuscript under consideration. Here in
this work, we investigate the role of reviewers
sentiments embedded within peer review texts
to predict the peer review outcome. Our pro-
posed deep neural architecture takes into ac-
count three channels of information: the pa-
per, the corresponding reviews, and the review
polarity to predict the overall recommenda-
tion score as well as the final decision. We
achieve significant performance improvement
over the baselines (∼ 29% error reduction)
proposed in a recently released dataset of peer
reviews. An AI of this kind could assist the ed-
itors/program chairs as an additional layer of
confidence in the final decision making, espe-
cially when non-responding/missing reviewers
are frequent in present day peer review."""


# instantiate the flask app
app = Flask(__name__)

# define the path
class CFG:
    DIR = os.getcwd()

stanfordnlp.download('en', CFG.DIR, force=True) # Download the English models

def read_pdf(file):
    '''
    This function takes in the input of the PDF file and 
    returns the raw text. Which has to go through pre-processing. 
    '''
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text().rstrip() + " "
    return text


def pre_process(text):
    '''
    This function takes applies two level of pre-processing
    - It removes the symbol "-\n" at the end of every sentence
    - It then uses the sentence tokenizer of standfordnlp and
    returns sentences which can be fed to the model.
    '''
    # corpus = text.splitlines()
    combined_text = text

    ## find a better, recursive way to do this ####
    # for i,_ in enumerate(corpus):
    #     if corpus[i][-1]=="-":
    #         corrected_string =  corpus[i] + corpus[i+1]
    #     else:
    #         corrected_string = corpus[i]
    #     combined_text+=corrected_string+"\n"
    #####################################

    nlp = stanfordnlp.Pipeline(processors='tokenize', models_dir=CFG.DIR, lang='en')
    document = nlp(combined_text)
    pre_processed_string = ""
    for _, sentence in enumerate(document.sentences):
        sent = ' '.join(word.text for word in sentence.words)
        pre_processed_string+=sent+'\n'

    return pre_processed_string.splitlines()

def infer(pre_processed_string):
    '''
    This code is used for inferring the extracted lines 
    from the PDF to find out the contributing statements.
    All items having 0 as the argmax are contributing
    statements.
    '''
    final_abstract = ""
    for extracted_sentence in pre_processed_string:
        inputs = tokenizer(extracted_sentence, padding=True, truncation=True, return_tensors="pt")
        logits = model(**inputs).logits
        if(argmax(logits).item()==0):
            final_abstract+=extracted_sentence+"\n"
    return final_abstract

@app.route('/')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        # get the file
        f = request.files['file']
        # read the text
        text = read_pdf(f)
        # pre-process the text
        pre_processed_string = pre_process(text)
        # infer and get the abstract summary
        final_abstract = infer(pre_processed_string)
        # calculate the rogue score
        rouge = ROUGEScore()
        rogue_scores = rouge(final_abstract, ground_truth)
        # convert the rougue score values to string for json response
        scores=[]
        for key in rogue_scores.keys():
            scores.append(str(key)+" is: "+str(rogue_scores[key].item()))
        # return the JSON response
        results = {"Text Summary": final_abstract, "Ground Truth":ground_truth, "Scores": scores}
        return jsonify(results)

if __name__ == '__main__':
   app.run() # removed debug = True