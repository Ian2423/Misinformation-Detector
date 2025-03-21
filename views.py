from django.views.decorators.csrf import csrf_exempt
import pickle
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack


from django.shortcuts import render
claim_result={}


# import logging
# logging.basicConfig(level=logging.ERROR)
def predict(request):
    if request.method == 'POST':
        post_claim = request.POST
        print(type(post_claim['claim']))
        claim_result['original_claim']=post_claim['claim']
        # Load the machine learning model
        with open('nlp/static/model.pkl', 'rb') as f:
            model = pickle.load(f)
        # print(model)
        # Get data from the request
        # data = misinformation_detector("Oliver Reed was a film actor.",model)

        file_path = "nlp/static/train.jsonl"
        data = read_fever_data(file_path)
        knowledge_graph = generate_knowledge_graph(data)
        refutes_count = number_of_refutes_edges(knowledge_graph)
        supports_count = number_of_supports_edges(knowledge_graph)

        print("Nodes in the knowledge graph:", knowledge_graph.number_of_nodes())
        print("Edges in the knowledge graph:", knowledge_graph.number_of_edges())
        print("The number of nodes with a REFUTES relationship is:", refutes_count)
        print("The number of nodes with a SUPPORTS relationship is:", supports_count)
        #dataset = prep_dataset(knowledge_graph)
        with open('nlp/static/dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        claim_features, evidence_features, labels, vectorizer = extract_features(dataset)
        X = hstack([claim_features, evidence_features])
        y = [1 if label == "SUPPORTS" else 0 for label in labels]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_prediction = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_prediction))
        print("Precision:", precision_score(y_test, y_prediction))
        print("Recall:", recall_score(y_test, y_prediction))
        print("F1 Score:", f1_score(y_test, y_prediction))
        final_result = misinformation_detector(post_claim['claim'],knowledge_graph, model, vectorizer)

        print(final_result)
        context={'result':final_result}

   

        return render(request,'display.html', context=context)
    else:
        return render(request,'display.html')

def read_fever_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
def generate_knowledge_graph(data):
    kgraph = nx.DiGraph()
    
#FEVER dataset consists of
#id: The claim ID
#claim: The claim's text.
#label: The label of the claim. This is either SUPPORTS, REFUTES or NOT ENOUGH INFO.
#evidence: There are evidence sets consisting of [Annotation ID, Evidence ID, Wikipedia URL, sentence ID] 
#or a [Annotation ID, Evidence ID, null, null] if the label is NOT ENOUGH INFO to support or refute the claim
    
    for item in data:
        claim_id = item['id']
        claim_text = item['claim']
        label = item['label']
                
#The nodes of the knowledge graph consisting of claims and evidences.
#The edges connecting the nodes and evidences contains a label providing information whether the evidence 
#supports or refutes the claim.
#The non verifiable claims have been ommited while generating the Knowledge Graph.
        
        # Adding the claim node
        kgraph.add_node(claim_id, label="claim", text=claim_text)
        
        if label != "NOT ENOUGH INFO":
            for evidence_group in item['evidence']:
                for evidence in evidence_group:
                    evidence_id = evidence[1]
                    evidence_title = evidence[2]
                    evidence_sentence_num = evidence[3]
                    
                    # Adding the evidence node
                    kgraph.add_node(evidence_id, label="evidence", title=evidence_title, sentence_num=evidence_sentence_num)
                    
                    # Adding an edge between claim and evidence with their relationship label
                    kgraph.add_edge(claim_id, evidence_id, label=label)
    
    return kgraph
def number_of_refutes_edges(kgraph):
    #This initializes the count variable to zero
    #It iterates over the edges in the knowledge graph, 
    count = 0
    for _, _, edge_data in kgraph.edges(data=True):
        if edge_data['label'] == 'REFUTES':
          count+=1
    #retrieving the edge data for each edge
    #If the label of the edge data is 'REFUTES', it increments the count.
    return count
    
    #It then outputs the the total count of edges with 'REFUTES' 
    #relationship in the knowledge graph.
def number_of_supports_edges(kgraph):
    count = 0
    for _, _, edge_data in kgraph.edges(data=True):
        if edge_data['label'] == 'SUPPORTS':
          count+=1
    #retrieving the edge data for each edge
    #If the label of the edge data is 'SUPPORTS', it increments the count.
    return count   
def text_preprocessor(text):
    tokens = nltk.word_tokenize(text.lower())
    #Tokenization is used in nltk.word_tokenize(text.lower()) to split the input text 
    #into individual words after converting the text to lowercase. 
    #These are tokens
    #Lowercasing ensures that the same word in different cases is treated 
    #as the same token.
    
    tokens = [token for token in tokens if token.isalpha()]
    #token.isalpha() removes non-alphabetic characters
    #Filtering out tokens that are not purely alphabetic like punctuation 
    #and numbers.
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    #stop_words = set(stopwords.words('english'))retrieves a set of English stop words from NLTK's corpus.
    #Removing common English stop words such as "the", "is", "in", from the tokens list.
    lemmatizer = WordNetLemmatizer()
    #Initializing the NLTK WordNet lemmatizer
    #Converts each token to its base or dictionary form, "running" becomes "run" etc
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)
    #The output combines the processed tokens back into a single string, with tokens separated by spaces.
def top_k_evidences(kgraph, claim_id, k=5):
    edges = [(evidence_id, data['label']) for _, evidence_id, data in kgraph.out_edges(claim_id, data=True)]
    evidences = sorted(edges, key=lambda x: x[1], reverse=True)[:k]
    return evidences

    #This iterates over the nodes in the graph, filtering for those labeled 
    #as 'claim'   
def prep_dataset(kgraph):
    dataset = []
    for claim_id, data in kgraph.nodes(data=True):
        if data['label'] == 'claim':
            claim_text = text_preprocessor(data['text'])
            #Preprocesses the claim text 
            evidences = top_k_evidences(kgraph, claim_id)
            #Calls top_k_evidences to find the top related evidences for each claim
            for evidence_id, relationship in evidences:
                evidence_data = kgraph.nodes[evidence_id]
                evidence_text = text_preprocessor(evidence_data['title'])
                 #Preprocesses the evidence text 
                dataset.append((claim_text, evidence_text, relationship))
                #Makes an ordered sequence of values of the preprocessed claim text, evidence text, and the relationship label to the dataset.

    return dataset
def extract_features(dataset):
    vectorizer = TfidfVectorizer()
    claims, evidences, labels = zip(*dataset)
    #This will transform textual data into a numerical format using the TF-IDF method
    #Initializes a TF-IDF vectorizer from the scikit-learn library to convert a collection of raw documents into a matrix 
    #of TF-IDF features.
    #Unpacks the dataset into separate lists of claims, evidences, and labels. 
    claim_features = vectorizer.fit_transform(claims)
    #Fits the vectorizer to the claims and transforms the claims into a matrix of TF-IDF features.
    evidence_features = vectorizer.transform(evidences)
    #Transforms the evidences into a matrix of TF-IDF features using the same vectorizer. To ensure that the feature space is consistent across both claims and evidences.
    return claim_features, evidence_features, labels, vectorizer
    #This returns the TF-IDF feature matrices for both claims and evidences, the labels, and the vectorizer itself.
def predict_claim_accuracy(claim_text, kgraph, model, vectorizer):
    claim_id = claims(kgraph, claim_text)
    if claim_id is None:
        print("This claim not found in the knowledge graph.")
        claim_result['claim']='This claim not found in the knowledge graph.'
        return None,None
#Searches the knowledge graph for a claim that exactly matches the given claim_text. 
#It returns the ID of the claim if found, or None if not found.
    claim = text_preprocessor(claim_text)
    print("Claim:", claim)
    claim_features = vectorizer.transform([claim])
    #print("Claim features shape:", claim_features.shape)

    get_top_k_evidences = top_k_evidences(kgraph, claim_id)
    prob = []
    predictions = []
    for evidence_id, relationship in get_top_k_evidences:
        evidence_data = kgraph.nodes[evidence_id]
        evidence_text = evidence_data['title']
        evidence_features = vectorizer.transform([evidence_text])
        features = hstack([claim_features, evidence_features])
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        if prediction[0] == 1:
            predicted_relationship = "SUPPORTS"
        else:
            predicted_relationship = "REFUTES"
        confidence = max(probability[0])
        predictions.append((predicted_relationship, evidence_text, confidence))
        print("Probability:", probability)
        prob.append(("Probability:", probability))
        #The predictions along with their probabilities and the corresponding evidence texts are stored in a list.
    return predictions, get_top_k_evidences
        #returns the list of predictions and the retrieved evidence.
def claims(kgraph, claim_text):
    for node_id, data in kgraph.nodes(data=True):
        if data['label'] == 'claim' and data['text'] == claim_text:
            return node_id
    return None
def misinformation_detector(claim_text,knowledge_graph, model, vectorizer):
    x=0
    predictions, get_top_k_evidences = predict_claim_accuracy(claim_text, knowledge_graph, model, vectorizer)
    if predictions is not None:
        supports_count = 0
        refutes_count = 0
        for predicted_relationship, evidence_text, confidence in predictions:
            print(f"{predicted_relationship} with evidence: {evidence_text} and confidence: {confidence:.2f}")
            claim_result[x]=f"{predicted_relationship} with evidence: {evidence_text} and confidence: {confidence:.2f}"
            x=x+1
            if predicted_relationship == "SUPPORTS":
                supports_count += 1
            elif predicted_relationship == "REFUTES":
                refutes_count += 1
        if supports_count > refutes_count:
            print("This claim is valid.")
            claim_result['claim']='This claim is valid.'
        else:
            print("This claim is invalid.")  
            claim_result['claim']='This claim is invalid.'
    
    else:
        print("There were no predictions made for the claim.")
        claim_result[0]=''
        claim_result[1]=''
        claim_result[2]=''
        claim_result[3]=''
        claim_result[4]=''
        claim_result['claim']='There were no predictions made for the claim.'
    print(claim_result)
    return claim_result

