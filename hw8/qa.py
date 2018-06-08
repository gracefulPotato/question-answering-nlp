# HW 8
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import re, nltk
from nltk.corpus import stopwords
import itertools
from nltk.stem.wordnet import WordNetLemmatizer
import math
import csv
from collections import defaultdict
from nltk.corpus import wordnet as wn
import string

should_normalize = True
DATA_DIR = "./wordnet"  # From wordnet_demo.py

def diagnose_goal(question_text,dep_q,par_q):
    #print(par_q)
    # Get question type (who, what, where, when, or why)                                                              
    question_type_who = re.match(r'[wW]ho (.*)',   question_text)
    question_type_what = re.match(r'[wW]hat (.*)',  question_text)
    question_type_where = re.match(r'[wW]here (.*)', question_text)
    question_type_when = re.match(r'[wW]hen (.*)',  question_text)
    question_type_why = re.match(r'[wW]hy (.*)',   question_text)
    question_type_how = re.match(r'[hH]ow [^lL](.*)', question_text)
    question_type_how_long = re.match(r'[hH]ow long (.*)', question_text)
    question_type_had = re.match(r'[hH]ad (.*)', question_text)
    question_type_did = re.match(r'[dD]id (.*)', question_text)
    if question_type_where:
        return ["PP"]
    elif question_type_who:
        return ["NP"]
    elif question_type_what:
        #check if it's a What do? question --> wants a verb
        if re.findall(r'[\n.]*\(VP\s*\(VBP? do\).*',str(par_q)):
            #print("DO QUESTIOHN")
            return ["VP"]
        #if quesition ends in preposition, it wants a PP answer
        if dep_q.nodes[len(question_text.split(" "))]["tag"] == "IN":
            print("WHAT PP QUESTION")
            return ["PP","NP"]
        return ["NP","ADJP"]
    elif question_type_why:
        return ["S","SBAR"]
    elif question_type_when:
        return ["NP"]
    elif question_type_how:
        return ["VBN","NP"]
    elif question_type_had:
        return ["RB"]
    elif question_type_how_long:
        return ["NP"]
    elif question_type_did:
        return ["NP"]
    else:
        return ["NP","AP"]

#From https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
def uniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  return output

def rec_print_vps(vp_structs):
    all_vps = []
    for struct in vp_structs:
        for item in struct:
            if "(VP" in str(item):
                for child in item:
                    if "(VP" in str(child):
                        print("item-child: "+str(len(child))+" "+str(child))
                        str_child = re.sub(r'\([A-Z]+','',str(child))
                        str_child = re.sub(r'\)','',str_child)
                        str_child = re.sub(r'\s',' ',str_child)
                        str_child = re.sub(r'\$','',str_child)
                        str_child = re.sub(r'\s+',' ',str_child)
                        str_child = re.sub(r'^ ','',str_child)
                        all_vps += str_child.split("and")
                        vps,vp_structs = check_for_top_pos(child,["VP"])
                        all_vps+=rec_print_vps(vp_structs)
    return all_vps
def create_discourse_model(par_s):
    discourse_model = []
    nouns_so_far = []
    locations_so_far = {}
    actions_so_far = {}
    animate_pronouns = ["he","He","she","She","her","Her","him","Him","his","His","her","Her"] #need to fix possesives
    for sentence in par_s:
        pronoun_map = {}
        nouns_in_sentence = mod_rec_check_for_pos(sentence,["NP"])
        #remove stopwords
        #determiner = ""
        #for noun in nouns_in_sentence:
        #    if noun.lower() in stopwords.words():
        #        nouns_in_sentence.remove(noun)
        #        determiner = noun
        #    else:
        #        if determiner != "":
        #            nouns_in_sentence.remove(noun)
        #            noun = determiner+" "+noun
        #            nouns_in_sentence.append(noun)
        #        determiner = ""
        print("NOUNS IN SENT: "+str(nouns_in_sentence))
        vps_in_sentence = rec_check_for_pos(sentence,["VP"])
        
        print("vps_in_sentence: "+str(vps_in_sentence))
        #while(r'\w' in str(check_for_top_pos)
        print(check_for_top_pos(sentence,["VP"]))
        vps,vp_structs = check_for_top_pos(sentence,["VP"])
        verb_phrases = rec_print_vps(vp_structs)
        print(verb_phrases)
        #for struct in vp_structs:
        #    for item in struct:
        #        if "VP" in str(item):
        #            for child in item:
        #                print("item-child: "+str(len(child))+" "+str(child))
        #                print(check_for_top_pos(child,["VP"]))
            #print(check_for_top_pos(struct,["VP"]))
        actions_so_far[par_s.index(sentence)] = verb_phrases #vps_in_sentence
        #print("nouns_in_sentence: "+str(nouns_in_sentence))
        #check for pronouns in sentence
        pronouns = rec_check_for_pos(sentence,["PRP"])
        #print("pronouns: "+str(pronouns))
        for pronoun in pronouns:
            if pronoun=="it":
                candidates = []
                for noun in nouns_so_far:
                    if re.match(r'([a-z\-]+)|([A-Z][a-z\-]+)',noun):
                        candidates.append(noun)
                pronoun_map[pronoun] = candidates
            elif pronoun in animate_pronouns:
                candidates = []
                for noun in nouns_so_far:
                    if re.match(r'(?:[A-Z][a-z\-]+)|(?:[a-z]+man)',noun):
                        candidates.append(noun)
                pronoun_map[pronoun] = candidates
            else:
                pronoun_map[pronoun] = nouns_so_far
            if len(pronoun_map[pronoun]) == 0:
                pronoun_map[pronoun] = nouns_in_sentence
        nouns_so_far += nouns_in_sentence
        nouns_so_far = uniq(nouns_so_far)
        discourse_model.append(pronoun_map)
        #add locations to locations_so_far
        pps = rec_check_for_pos(sentence,["PP"])
        print("pps: "+str(pps))
        #for pp in pps:
        if len(pps)>0 and ((pps[0] == "in" and pps[1]!="order") or pps[0] == "at" or pps[0] == "into" or pps[0] == "on"):
            locations_so_far[par_s.index(sentence)] = (" ".join(pps))
        if len(pps)>0 and pps[0] == "by":
            locations_so_far[par_s.index(sentence)] = (" ".join(pps))
    print("LOCATILNS: "+str(locations_so_far))
    return discourse_model, locations_so_far, actions_so_far

def normalize_verb(keywords,dep_q):
    #need to normalize verb (unless it's a stopword like be)
    for node in dep_q.nodes:
        if "VB" in dep_q.nodes[node]["tag"]:
            if node-1 in range(len(keywords)):
                keywords[node-1] = "[^\.]*"+dep_q.nodes[node]["lemma"]+"[^\ ]*"
    return keywords

def move_auxiliaries(keywords,dep_q):
    #find the verb's index
    verb_index = 0
    for node in range(len(dep_q.nodes)-1,0,-1):
        if "VB" in dep_q.nodes[node]["tag"]:
            if dep_q.nodes[node-1]["tag"]=="RB":
                verb_index = node-3
            else:
                verb_index = node-2
            break
    #look for auxes and put them right before verb
    for node in dep_q.nodes:
        if dep_q.nodes[node]["rel"] == "aux":
            auxiliary = keywords[node-1]
            keywords.remove(auxiliary)
            keywords.insert(verb_index,auxiliary.lower())
            #increment the verb's index since it's pushed one later            
            verb_index = verb_index + 1
    return keywords

def rec_check_for_pos(par_q,posses):
    pos_instances = []
    for child in par_q:
        if child in par_q.leaves():
            return []
        elif any(pos in child.label() and not (pos == "VP" and child.label() == "ADVP") for pos in posses):
            pos_instances = pos_instances +child.leaves()
        else: 
            pos_instances = pos_instances + rec_check_for_pos(child,posses)
    return pos_instances

def mod_rec_check_for_pos(par_q,posses):
    pos_instances = []
    for child in par_q:
        if child in par_q.leaves():
            return []
        elif any(pos in child.label() and not (pos == "VP" and child.label() == "ADVP") for pos in posses):
            pos_instances = pos_instances +[" ".join(child.leaves())]
        else:
            pos_instances = pos_instances + rec_check_for_pos(child,posses)
    return pos_instances

def check_for_top_pos(par_q,posses):
    pos_instances = []
    sub_structures = []
    for child in par_q:
        if child in par_q.leaves():
            return [],[]
        elif any(pos == child.label() and not (pos == "VP" and child.label() == "ADVP") for pos in posses):
            pos_instances = pos_instances +child.leaves()
            sub_structures.append(child)
        else:
            child_pos,child_struct = check_for_top_pos(child,posses)
            pos_instances = pos_instances + child_pos
            sub_structures += child_struct
    return pos_instances,sub_structures

def remove_aux(question_text,dep_q):
    question_words = question_text.split()
    dep_q_index = 1
    for word in question_words:
        if dep_q.nodes[dep_q_index]["rel"]=="aux":
            question_words.remove(word)
        dep_q_index = dep_q_index+1
    return " ".join(question_words)

def remove_stopwords(words):
    for i in range(len(words)):
        if re.sub("[^\w]","",words[i]) in stopwords.words('english'):
            words[i] = "[^\.]*"
    return words

def permute_and_join(keywords):
    keyword_combos = list(itertools.permutations(keywords))
    all_keywords = "(?:"+re.sub("\?",""," ".join(keywords[1:]))+")"
    for keywords in keyword_combos:
        keyword = " ".join(keywords[1:])
        keyword = re.sub("\?","",keyword)
        all_keywords = all_keywords+"|(?:"+keyword+"[^\.]*)"
    return all_keywords

def get_keyword(question,dep_q):
    keywords = question["text"].split()
    #need to normalize verb (unless it's a stopword like be)

    if should_normalize:
        keywords = normalize_verb(keywords,dep_q)
    #move auxiliaries like "might" to be before the verb
    keywords = move_auxiliaries(keywords,dep_q)
    keywords = remove_stopwords(keywords)
    #permute keywords to account for different orderings
    if len(keywords) <= 6:    ### to make runtime faster
        keyword = permute_and_join(keywords)
    else:
        keyword = get_noun(question["text"],dep_q)
        #keyword = " ".join(rec_check_for_pos(question["par"],["NP"]))
    return keyword

# Get noun from question
def get_noun(question,dep_q):
    keywords = question.split()
    for node in dep_q.nodes:
        if node-1 in range(len(keywords)):
            if "NNP" or "NN" or "NNS" in dep_q.nodes[node]["tag"]:   # only gets last NN* in question
                keyword = dep_q.nodes[node]["lemma"]
            else:
                keyword = "none"
    keyword = re.sub('[?,.!]', '', keyword)
    return keyword

# Get verb from question                                                                                       
def get_verb(question,dep_q):
    keywords = nltk.word_tokenize(question)
    print("get_verb: "+str(keywords))
    keywords = question.split(" ")
    for node in dep_q.nodes:
        if node-1 in range(len(keywords)):
            if "VB" or "VBN" in dep_q.nodes[node]["tag"]:   # only gets last VB* in question 
                keyword = dep_q.nodes[node]["lemma"]
            else:
                keyword = "none"
    keyword = re.sub('[?,.!]', '', keyword)
    return keyword

#def get_verb_con(question,par_q):
    

# Get all words with parts of speech in the pos_list from answer sentence
def get_answer_pos(best_sent_index, dep_s, pos_list):
    words = []
    for x,y in dep_s[best_sent_index].nodes.items():
        tag = y["tag"]
        if any(pos in str(tag) for pos in pos_list):
            words.append(y["word"])

    if len(words) > 0:
        answer = ' '.join(words)
    else:
        answer = "it"

    return answer

# Get all nouns from answer sentence
def get_answer_noun(best_sent_index, dep_s):
    nouns= []
    for x,y in dep_s[best_sent_index].nodes.items():
        tag = y["tag"]
        if "NN" in str(tag):
            nouns.append(y["word"])
    
    if len(nouns) > 0:
        answer = ' '.join(nouns)
    else:
        answer = "it"

    return answer

def get_answer_nsubj(best_sent_index, dep_s):
    nsubj = "nsubj"
    for x, y in dep_s[best_sent_index].nodes.items():
        tag = y["rel"]
        if "nsubj" in str(tag):
            nsubj = y["word"]
    return nsubj

def resolve_pronouns(sentence,sent_discourse_model):

    lmtzr = WordNetLemmatizer()
    words =  nltk.word_tokenize(sentence)
    resolved_words = []
    for word in words:
        if word in sent_discourse_model:
            mapping = sent_discourse_model[word]
            word = [mapping[0]]
        else:
            word = [word]
        resolved_words += word
    return resolved_words

# Look for negations in both question and answer
def yes_no_q(question_text, story, goal_constituents, discourse_model):

    best_sent, best_index = question_answer_similarity(question_text, story, goal_constituents, discourse_model)
    question_words = nltk.word_tokenize(question_text)
    answer_words = nltk.word_tokenize(best_sent)

    should_negate = False
    for word in question_words:
        if word == ("not", "didn't", "hadn't"):
            print(word)
            should_negate = True
    
    answer = "yes"
    for word in answer_words:
        if word == ("not", "didn't" , "hadn't"):
            answer = "no"
        else:
            answer = "yes"

    if should_negate:
        if answer == "yes":
            answer = "no"
        else:
            answer == "yes"
    return answer

# From wordnet_demo.py
def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename:
        type = "noun"
    else:
        type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        word_ids[line['synset_id']] = {'synset_offset': line['synset_offset'],
                                       'story_'+type: line['story_'+type], 'stories': line['stories']}
    return word_ids

# Uses code from wordnet_demo.py
def get_story_nouns_verbs(story, sentence):
    s_id = story["sid"]

    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

    nouns = []
    for synset_id, items in noun_ids.items():
        noun = items['story_noun']
        stories = items['stories']
        if len(stories) >= 1 and s_id+".vgl" in stories:
            nouns.append(noun)

    verbs = []
    for synset_id, items in verb_ids.items():
        verb = items['story_verb']
        stories = items['stories']
        if len(stories) >= 1 and s_id+".vgl" in stories:
            verbs.append(verb)
    # print(nouns + verbs)
    return nouns + verbs

# Uses code from wordnet_demo.py
def get_story_wn_ids(story, sentence):
    s_id = story["sid"]
    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

    nouns = []
    for synset_id, items in noun_ids.items():
        noun_id = synset_id
        stories = items['stories']
        if len(stories) >= 1 and s_id+".vgl" in stories:
            nouns.append(noun_id)

    verbs = []
    for synset_id, items in verb_ids.items():
        verb_id = synset_id
        stories = items['stories']
        if len(stories) >= 1 and s_id+".vgl" in stories:
            verbs.append(verb_id)
    print(nouns + verbs)
    return nouns + verbs

def get_full_synset(story, sentence):
    # noun_verb_ids = get_story_nouns_verbs(story, sentence)
    noun_verb_ids = get_story_wn_ids(story, sentence)
    all_synonyms = []

    for an_id in noun_verb_ids:
        # print(wn.synset(an_id))
        # for i in get_synsets(an_id):
        all_synonyms.append(wn.synset(an_id))
    return all_synonyms

# Old version
# def get_wn_overlap(question, story):
#     q_id = question["qid"]
#     q_text = question["text"]
#     s_id = question["sid"]
#     s_text = story["text"]
#     q_words = nltk.word_tokenize(q_text)

#     # Remove stopwords from question
#     q_synonyms = []
#     first = True
#     for word in q_words:
        
#         if first and word not in stopwords.words('English'):
#             q_words.remove(word)
#             first = False
        
#         if word in stopwords.words('English') or word == '?':
#             q_words.remove(word)
#         # else:
#             # syns = get_synonyms(word)
#             # print("WORD TO GET: {}".format(word))
#             # if len(syns) >= 1:
#             #     for s in syns:
#             #         q_synonyms.append(s)
    
#     # for word in q_words:
#     #     syns = get_synonyms(word)
#     #     # print("WORD TO GET: {}".format(word))
#     #     if len(syns) >= 1:
#     #         for s in syns:
#     #             q_synonyms.append(s)
    
#     # print("        qwords: {}".format(q_words))
#     # q_words_all = q_words + q_synonyms
#     # s_all_synonyms = get_full_synset(story)
#     text_freq = {}
#     sent_index = 0

#     # print("  qwordsall: {}\n".format(q_words_all))
#     for s_sent in nltk.sent_tokenize(s_text):
#         # s_all_synonyms = get_full_synset(story, s_sent)
#         for s_word in s_sent:
#             s_all_synonyms = get_hypo_hyper(s_word)
#         text_freq[s_sent] = 0
#         # print("q_words {}".format(q_words))
#         for q_word in q_words:
#             # print("q word: {}".format(q_word))
#             if q_word in s_all_synonyms:
#                 # print("dict item: {}".format(s_sent))
#                 text_freq[s_sent] += 1
#         sent_index += 1
#         # print("next sent")
#         # print(s_all_synonyms)
#     # print("        sallsynonyms: {}\n".format(s_all_synonyms))
#     best_sentence = max(text_freq, key=text_freq.get)
#     best_index = nltk.sent_tokenize(s_text).index(best_sentence)
#     # print("best sentence: {}".format(best_sentence))
#     # print(text_freq)

#     return best_sentence, best_index

# Updated version
def get_wn_sent_overlap(question, story):
    quest_text = question["text"]
    story_text = story["text"]

    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))

    s_sents = nltk.sent_tokenize(story_text)
    q_words = nltk.word_tokenize(quest_text)
    max_value = 0

    # Start by normalizing the question
    q_words = q_words[1:-1]           # Citation: https://stackoverflow.com/questions/11338143/how-to-remove-the-first-and-last-item-in-a-list
    quest_words = []
    for w in q_words:
        if w not in stopwords.words('english'):
            quest_words.append(w)
    
    # 1. Start by trying to find the best sentence by matching the question directly
    # 2. If no matches then try lemmatizing the words in both the question and the text
    # 3. If no matches then try finding synonyms with WordNet
    s_sents = nltk.sent_tokenize(story_text)
    s_sentences = {}
    s_index = 0
    for s_sent in s_sents:
        # print("sentence: {}".format(s_sent))
        s_index += 1
        s_sentences[s_sent] = 0
        s_sent_synonyms = []
        s_words = nltk.word_tokenize(s_sent)
        
        for s_word in s_words:
            synsets = wn.synsets(s_word)
            for syn in synsets:
                if syn.name() in noun_ids or syn.name() in verb_ids:
                    for s in syn.lemmas():
                        if s.name() not in s_sent_synonyms:
                            s_sent_synonyms.append(s.name())
                        if s_word not in s_sent_synonyms:
                            s_sent_synonyms.append(s_word)   # also append the word itself as a backup
                        # print(s.name())
                        if s.name() in q_words:
                            # print("...sentence: {}".format(s_words))
                            # print("...sMATCH: {}, {}.".format(s.name(), q_words))
                            # print("...synonyms: {}\n".format(s_sent_synonyms))
                            s_sentences[s_sent] += 1
                        else:   # if you can't find a match, try lemmatizing words in question
                            lmtzr = WordNetLemmatizer()
                            lems = []
                            for w in q_words:
                                lem = lmtzr.lemmatize(w, 'v')
                                lems.append(lem)
                            if s.name() in lems:
                                # print("...sentence: {}".format(s_words))
                                # print("...lMATCH: {}, {}.".format(s.name(), q_words))
                                # print("...synonyms: {}\n".format(s_sent_synonyms))
                                s_sentences[s_sent] += 1
                            # print(s.name())
                else:
                    continue
        
                
    best_sentence = max(s_sentences, key=s_sentences.get)
    best_index = s_sents.index(best_sentence)

    m = max(s_sentences, key=s_sentences.get)
    max_value = s_sentences[m]

    # 4. If no synonyms found, try matching hypernyms and hyponyms:
    if max_value < 1:
        text_freq = {}
        index = 0
        for s_sent in s_sents:
            index += 1
            text_freq[s_sent] = 0
            s_sent_hyper = []
            s_sent_hypo = []
            s_words = nltk.word_tokenize(s_sent)
            for s_word in s_words:
                if s_word.lower() not in stopwords.words('english'):
                    synsets = wn.synsets(s_word)
                    for s in synsets:
                        if s.name() in noun_ids or s.name() in verb_ids:
                            for h in s.hypernyms():
                                lemmas = h.lemmas()
                                for l in lemmas:
                                    s_sent_hyper.append(l.name())
                                    if l.name() in q_words:
                                        # print("...sentence: {}".format(s_words))
                                        # print("...heMATCH: {}, {}.".format(s.name(), q_words))
                                        text_freq[s_sent] += 1
                            for h in s.hyponyms():
                                lemmas = h.lemmas()
                                for l in lemmas:
                                    s_sent_hypo.append(l.name())
                                    if l.name() in q_words:
                                        # print("...sentence: {}".format(s_words))
                                        # print("...hoMATCH: {}, {}.".format(s.name(), q_words))
                                        text_freq[s_sent] += 1
        m = max(text_freq, key=text_freq.get)
        max_val = text_freq[m]
        best_sentence = max(text_freq, key=text_freq.get)
        best_index = s_sents.index(best_sentence)

    return best_sentence, best_index

def reformulate_question_wn(question, story):
    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))
    q_text = question["text"]
    q_words = nltk.word_tokenize(q_text)
    q_words = q_words[1:-1]  # remove question word and question mark
    words = []
    for word in q_words:
        if word not in stopwords.words('english'):
            words.append(word)
    
    synonyms = []
    for w in words:
        syns = wn.synsets(w)
        for syn in syns:
            if syn.name() in noun_ids or syn.name() in verb_ids:
                syn = syn.lemma_names()
                for s in syn:
                    if s not in synonyms:
                        synonyms.append(s)

# Uses code from wordnet_demo.py
def get_synsets(word_id):
    return wn.synsets(word_id)

def get_hypo_hyper(word):
    return get_hyponyms(word) + get_hypernyms(word)

# Uses code from wordnet_demo.py
def get_hyponyms(word):
    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))
    all_hyponyms = []
    for synset in get_synsets(word):
        hyponyms = synset.hyponyms()
        # print("%s: %s" % (synset, hyponyms))
        for hypo in hyponyms:
            # print(hypo.name()[0:hypo.name().index(".")])
            if hypo.name() in noun_ids or hypo.name() in verb_ids: 
                all_hyponyms.append(hypo.name()[0:hypo.name().index(".")])
                # print(hypo.name())
    # print("all hypo: \n{}\n".format(all_hyponyms))
    return all_hyponyms

# Uses code from wordnet_demo.py
def get_hypernyms(word):
    noun_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_nouns.csv"))
    verb_ids = load_wordnet_ids("{}/{}".format(DATA_DIR, "Wordnet_verbs.csv"))
    all_hypernyms = []
    for synset in get_synsets(word):
        hypernyms = synset.hypernyms()
        for hyper in hypernyms:
            if hyper.name() in noun_ids or hyper.name() in verb_ids:
                all_hypernyms.append(hyper.name()[0:hyper.name().index(".")])
    # print("all hyper: \n{}\n".format(all_hypernyms))
    return all_hypernyms

# Match the question with the sentence with the most overlapping words
def question_answer_similarity(question_text, story, goal_constituents,discourse_model):
    story_text = story["text"]
    story_par = story["story_par"]
    story_dep = story["story_dep"]
    sch_text = story["sch"]
    sch_par = story["sch_par"]
    question_words = nltk.word_tokenize(question_text)
    #print("QUESTIOJ WOKRD: "+str(question_words))
    text_sentences = nltk.sent_tokenize(story_text)
    text_freq = {}
    lmtzr = WordNetLemmatizer()
    sent_index = 0
    for sentence in text_sentences:
        if sent_index >= len(discourse_model) or sent_index >= len(story_dep):
            break
        text_words = resolve_pronouns(sentence,discourse_model[sent_index])
        lemma_text_words = []
        for word in text_words:
            lemma_text_words.append(lmtzr.lemmatize(word,'v'))
#        lemma_text_words = [lmtzr.lemmatize(word) for word in text_words]
        #print("LEMMA TEXT WORDS: "+str(lemma_text_words))
        text_freq[sentence] = 0
        for word in question_words:
            if word in text_words and word not in stopwords.words('english'):
                #print("Word: "+word+" in textwrods")
                text_freq[sentence] += 1
                #weight it more heavily if it's a verb
                for node in story_dep[sent_index].nodes:
                    if story_dep[sent_index].nodes[node]["word"]==word and "VB" in story_dep[sent_index].nodes[node]["tag"]:
                        text_freq[sentence]+=1
            if lmtzr.lemmatize(word,'v') in lemma_text_words and word not in stopwords.words('english'):
                #print("Word: "+word+"'s lemma "+lmtzr.lemmatize(word)+" in textwrods")
                text_freq[sentence] += 1
                for node in story_dep[sent_index].nodes:

                    story_lemma = lmtzr.lemmatize(str(story_dep[sent_index].nodes[node]["word"]))
                    if story_lemma ==lmtzr.lemmatize(word) and "VB" in story_dep[sent_index].nodes[node]["tag"]:
                        text_freq[sentence]+=2
            #else:
                #print("Word: "+word+"'s lemma "+lmtzr.lemmatize(word)+" not in textwrods")
#        print("text_sentences.index(sentence): "+str(text_sentences.index(sentence)%(len(story_par)-1))+" and len(tory_par): "+str(len(story_par)))

        if not any(pos in str(story_par[text_sentences.index(sentence)%(len(story_par)-1)]) for pos in goal_constituents):
            text_freq[sentence] -= 10
        sent_index += 1
    print(text_freq)
    best_sentence = max(text_freq, key=text_freq.get)
    best_index = text_sentences.index(best_sentence)

    return best_sentence, best_index

# Finds the match that is most similar to the question
def best_match(question_text, matches):
    question_words = nltk.word_tokenize(question_text)
    text_freq = {}
    for match in matches:
        match_string = nltk.word_tokenize(match)
        text_freq[match] = 0
        for word in question_words:
            if word in match_string and word not in stopwords.words('english'):
                text_freq[match] += 1
    
    best_match = max(text_freq, key=text_freq.get)
    return best_match        

# Make question into declarative statement
def normalize_question(question_text,dep_q,constituency_parse):   
    # Remove the auxiliary
    question_text = remove_aux(question_text,dep_q)
    # Remove initial question word
    who = re.match(r'[Ww]ho (.*)',   question_text)
    what = re.match(r'[Ww]hat (.*)',  question_text)
    where = re.match(r'[Ww]here (.*)', question_text)
    when = re.match(r'[Ww]hen (.*)',  question_text)
    why = re.match(r'[Ww]hy (.*)',   question_text)
    how = re.match(r'[Hh]ow (.*)', question_text)
    if who:
        question_text = who.group(1)
        # print("normalizing WHO: "+question_text)
    elif what:
        question_text = what.group(1)
    elif where:
        question_text = where.group(1)
    elif when:
        question_text = when.group(1)
    elif why:
        question_text = why.group(1)
    elif how:
        question_text = how.group(1)
    else:
        print(question_text)
    # Remove question mark
    question_text = re.match(r'(.*)\?', question_text)
    question_text = question_text.group(1)

    return question_text

def get_lemma(word):
    return WordNetLemmatizer().lemmatize(word, 'v')
                
def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    lmtzr = WordNetLemmatizer()

    text_q = question["text"]
    dep_q = question["dep"]
    par_q = question["par"]
    text_s = story["text"]
    dep_s = story["story_dep"]
    par_s = story["story_par"]

    print("get_verb: "+get_verb(text_q,dep_q))
    print(check_for_top_pos(par_q,["VP"]))
    print("reccheckforpos vpb"+str(rec_check_for_pos(par_q,["VB"])))

    discourse_model, location_model, action_model  = create_discourse_model(par_s)

    answer = None


    sch_discourse_model, sch_location_model, sch_action_model  = create_discourse_model(story["sch_par"])
    print("\n"+question["qid"])

    #print("discourse model: "+str(discourse_model))

    #diagnose what kind of constituent the question wants
    goal_constituents = diagnose_goal(text_q,dep_q,par_q)

    keyword = ""
    for node in dep_q.nodes:
        if dep_q.nodes[node]["rel"]=="nsubj":
            keyword = dep_q.nodes[node]["lemma"]
    numwords = len(text_q.split(" "))
    if keyword=="":
        for node in dep_q.nodes:
            if dep_q.nodes[node]["rel"]=="nobj":
                keyword = dep_q.nodes[node]["lemma"]
    if keyword=="":
        keyword = dep_q.nodes[numwords]["lemma"]

    poss_adj = rec_check_for_pos(par_q,["JJ","NN"]) 
    if poss_adj != None:
        keyword = "[^\.]*"+"[^\.]*".join(poss_adj)+"[^\.]"

    verb = rec_check_for_pos(par_q,["V"])
    #keyword = "[^\.]*"+"[^\.]*".join(verb)+"[^\.]"

    question_text = question.get("text")  # gets the raw text of the question
    question_difficulty = question.get("difficulty")

    yes_no_question = False

    # Get question type (who, what, where, when, or why)
    question_type_who = re.match(r'[wW]ho (.*)',   question_text)
    question_type_what = re.match(r'[wW]hat (.*)',  question_text)
    question_type_where = re.match(r'[wW]here (.*)', question_text)
    question_type_when = re.match(r'[wW]hen (.*)',  question_text)
    question_type_why = re.match(r'[wW]hy (.*)',   question_text)
    question_type_did = re.match(r'[dD]id (.*)', question_text)
    question_type_had = re.match(r'[hH]ad (.*)', question_text)
    question_type_how = re.match(r'[hH]ow (.*)',question_text)

    if question_difficulty == "Easy":
        if question_type_who:
            keyword = get_keyword(question,dep_q) #question_type_who.group(),dep_q)
            #print("WHO keyword: "+keyword)
        if question_type_what:
            keyword = get_noun(question_type_what.group(), dep_q)
            keyword += get_verb(question_type_what.group(), dep_q)
        if question_type_where:
            keyword = get_keyword(question,dep_q) #question_type_where.group(),dep_q)
        if question_type_when:
            keyword = get_noun(question_type_when.group(), dep_q)
        if question_type_why:
            keyword = get_noun(question_type_why.group(), dep_q)
        if question_type_had:
            yes_no_question = True
        if question_type_did:
            yes_no_question = True
        if question_type_how:
            keyword = get_keyword(question,dep_q) #question_type_how.group(),dep_q)

        global should_normalize
        should_normalize = True

        lmtzr = WordNetLemmatizer()
        story_words = nltk.word_tokenize(story["text"].lower())
        lemmad_words = []

        for word in story_words:
            lemmad_words.append(lmtzr.lemmatize(word))
        matches = re.findall(("[^\.]*"+keyword+"[^\.]*").lower(),story["text"].lower())

        if len(matches) != 0:
            if question_type_why:
                for match in matches:
                    sentence = re.findall("to|because", match)
                    if sentence:
                        answer = match
                        break
                    else:
                        continue
            elif question_type_who:  
                best = best_match(question_text, matches)
                best = best + '.'
                best = nltk.word_tokenize(best)
                tags = nltk.pos_tag(best)
                clean = []
                noun = r'NN.?.?'

                question_words = nltk.word_tokenize(question_text)
                previous_pair = None
                for pair in tags:
                    pos = re.findall(noun, pair[1])
                    if pos and pair[0] not in question_words and pair[0] not in clean:
                        if previous_pair != None and "DT" in previous_pair[1]:
                            clean.append(previous_pair[0])
                        clean.append(pair[0])
                    previous_pair = pair
                if len(clean)>0:
                    return " ".join(clean[:2])  # returning the first noun produced the best result
                else:
                    return "HALP"
            elif question_type_where:
                # Only use the best match
                best = best_match(question_text, matches)
                # print("     best match: {}".format(best))
                match_obj = re.match(r"[Ww]here (do|did|is|was) (.*)", question_text)
                found = False
                if match_obj:
                    found = True
                    decl_stmt = match_obj.group(2).replace("?", "")
                    if found:
                        sentence = re.match(decl_stmt, best)
                        if sentence:
                            sentence.group()
                            answer = sentence.group()
                        else:  # if you can't find it, try getting the lemmas of each word in question and answer sentence
                            q_words = nltk.word_tokenize(decl_stmt)
                            a_words = nltk.word_tokenize(best)
                            lem_words_q = []
                            lem_words_a = []
                            for q_word in q_words:
                                lem_words_q.append(get_lemma(q_word))
                            for a_word in a_words:
                                lem_words_a.append(get_lemma(a_word))

                            q_stmt = ' '.join(word for word in lem_words_q)
                            a_stmt = ' '.join(word for word in lem_words_a)
                            # print("     Q Stmt: {}".format(q_stmt))
                            # print("     A Stmt: {}".format(a_stmt))
                            lem_match = re.sub(q_stmt, '', a_stmt)
                            if lem_match:
                                answer = lem_match
                            else:
                                answer = best
                else:
                    answer = best
            else:
                answer = best_match(question_text, matches)    
        answer,sentence_index = question_answer_similarity(question_text, story ,goal_constituents,discourse_model)#[0]
        answer = " ".join(rec_check_for_pos(par_s[sentence_index],goal_constituents))
        sche_answer = ""
        if type(story["sch"]) is str:

            #print(nltk.sent_tokenize(story["sch"])[sentence_index])
            #print("STORY: "+answer+" VERSUS SCHERAAZAHD: "+nltk.sent_tokenize(story["sch"])[sentence_index])  
            sche_answer,sche_index = question_answer_similarity(question_text,story,goal_constituents,sch_discourse_model)
            # print("SCHEANSWER: "+sche_answer)
            sche_answer = " ".join(rec_check_for_pos(story["sch_par"][sche_index],goal_constituents))
        answer = best_match(question_text,[answer,sche_answer])
        #answer = get_answer_pos(sentence_index, dep_s, ["NN","JJ"]) #noun(sentence_index, dep_s)
        #answer = " ".join(rec_check_for_pos(par_s[sentence_index],goal_constituents))

    elif question_difficulty == "Medium":
        question_text = normalize_question(question_text,dep_q,par_q)
        #answer = question_answer_similarity(question_text, story)
        answer,sentence_index = question_answer_similarity(question_text, story,goal_constituents,discourse_model)#[0]
        # print("answer: "+answer)
        #sentence_index = question_answer_similarity(question_text, story)[1] #what is sentence_index?
        # answer = get_answer_noun(sentence_index, dep_s)   
        answer=" ".join(rec_check_for_pos(par_s[sentence_index],goal_constituents)).lower()

        if question_type_did:
            answer = yes_no_q(question_text, story, goal_constituents, discourse_model)

        sche_answer = ""
        if type(story["sch"]) is str:
     
            sche_answer,sche_index = question_answer_similarity(question_text,story,goal_constituents,sch_discourse_model)

            # print("SCHEANSWER: "+sche_answer)
            sche_answer = " ".join(rec_check_for_pos(story["sch_par"][sche_index],goal_constituents))
        answer = best_match(question_text,[answer,sche_answer])

    elif question_difficulty == "Hard":

        print("\n{} | {}".format(question["qid"], question_difficulty))
        print("-----QUESTION: {}".format(question_text))
        # answer = get_wn_overlap(question,story)[0]
        answer, sentence_index = question_answer_similarity(
                    question_text, story, goal_constituents, discourse_model)
        answer = " ".join(rec_check_for_pos(
            par_s[sentence_index], goal_constituents)).lower()
        answer = get_wn_sent_overlap(question, story)[0]
        print("-----ANSWER: {}".format(answer))
        # answer = question_answer_similarity(
        #     question_text, story, goal_constituents, discourse_model)[0]

    elif question_difficulty == "Discourse":
        answer = question_answer_similarity(
            question_text, story, goal_constituents, discourse_model)[0]

    
        print(question_difficulty)
        answer, sentence_index  = question_answer_similarity(
            question_text, story, goal_constituents, discourse_model)
        answer=" ".join(rec_check_for_pos(par_s[sentence_index],goal_constituents)).lower()
        if "where" in question_text or "Where" in question_text:
            i = 0
            if "after" in question_text:
                while sentence_index+i not in location_model:
                    i+=1
                answer = location_model[sentence_index+i]
            elif "before" in question_text:
                while sentence_index-i not in location_model:
                    i+=1
                answer =location_model[sentence_index+i]
            else:
                while i not in location_model:
                    i+=1
                answer =location_model[i]
            return answer
        if "why" in question_text or "Why" in question_text:
            answer=" ".join(rec_check_for_pos(par_s[sentence_index-1],goal_constituents)).lower()
        if "who" in question_text or "Who" in question_text:
            i = 0
            for sent in discourse_model:
                print(str(i)+" "+str(sent))
                i += 1
            answer = []
            for pronoun in rec_check_for_pos(par_s[sentence_index],goal_constituents):
                if pronoun in discourse_model[sentence_index]:
                    #prioritize capitalized rather than 0th
                    for antecedent in pronoun:
                        if antecedent[0].isupper() and antecedent[0] not in stopwords.words:
                            print(antecedent[0]+" not in "+str(stopwords.words))
                            answer.append(antecedent)
                            #break
                if answer == []:
                    answer.append(discourse_model[sentence_index][pronoun][0])
            if len(answer)>0:
                answer = " ".join(list(set(answer)))
            else:
                answer=" ".join(rec_check_for_pos(par_s[sentence_index],goal_constituents)).lower()
        if "what" in question_text or "What" in question_text:
            #return next in discourse_model list of actions
            #find the question's verb in the action_model
            question_verb = rec_check_for_pos(par_q,["VB"])#get_verb(question_text,dep_q) #rec_check_for_pos(question_text,["VB"])
            question_verb = question_verb[len(question_verb)-1]
            #print("qeustiontext: "+question_text+" and dep_q: "+str(dep_q))
            print("Searching for question_verb: "+str(question_verb)+" in action model: "+str(action_model[sentence_index]))
            question_verb = question_verb #[len(question_verb)-1]
            vp_index = 0
            for vp in action_model[sentence_index]:
                print("vp in action model:  "+vp)
                if question_verb in vp:
                    break
                else:
                    vp_index+=1
#            print("quesiton vp: "+str(action_model[sentence_index][vp_index]))
            if len(action_model[sentence_index])>vp_index+1:
                print("quesiton vp: "+str(action_model[sentence_index][vp_index]))
                answer = action_model[sentence_index][vp_index+1]
            else:
                j = 1
                while len(action_model[sentence_index+j])==0:
                    j+=1
                if j >= len(action_model):
                    while len(action_model[sentence_index+j])==0:
                        j-=1
                print("choosing first of "+str(action_model[sentence_index+j]))
                answer = action_model[sentence_index+j][-1]
            if "after" in question_text or "After" in question_text:
                vp_index = len(action_model[sentence_index])-1
                for i in range(len(action_model[sentence_index])-1,0,-1):
                #for vp in action_model[sentence_index]:
                    vp = action_model[sentence_index][i]
                    print("after vp in action model:  "+vp)
                    lemma_vp = ""
                    for word in nltk.word_tokenize(vp):
                        lemma_vp+=lmtzr.lemmatize(word,'v')
                        lemma_vp+=" "
                    if lmtzr.lemmatize(question_verb,'v') in lemma_vp:
                        print(lmtzr.lemmatize(question_verb,'v')+" in "+lemma_vp)
                        break
                    else:
                        print(lmtzr.lemmatize(question_verb,'v')+" not in "+lemma_vp)
                        vp_index-=1
                if len(action_model[sentence_index])>vp_index+1:
                    print("quesiton vp: "+str(action_model[sentence_index][vp_index]))
                    answer = action_model[sentence_index][vp_index+1]
                else:
                    j = 1
                    while len(action_model[sentence_index+j])==0:
                        j+=1
                    if j >=len(action_model):
                        while len(action_model[sentence_index+j])==0:
                            j-=1
                    print("choosing first of "+str(action_model[sentence_index+j]))
                    answer = action_model[sentence_index+j][0]
            elif "before" in question_text or "Before" in question_text:
                vp_index = 0
                for vp in action_model[sentence_index]:
                    print("before vp in action model:  "+vp)
                    if question_verb in vp:
                        break
                    else:
                        vp_index+=1
                if len(action_model[sentence_index])>vp_index:
                    print("quesiton vp: "+str(action_model[sentence_index][vp_index]))
                    answer = action_model[sentence_index][vp_index-1]
                else:
                    j = 1
                    while len(action_model[sentence_index-j])==0:
                        j+=1
                    if j >=len(action_model):
                        while len(action_model[sentence_index-j])==0:
                            j-=1
                    print("choosing last of "+str(action_model[sentence_index-j]))
                    answer = action_model[sentence_index-j][-1]
    else:
        answer = question_answer_similarity(question_text, story,goal_constituents,discourse_model)[0]
    if yes_no_question:
        if 'n' in answer:
            answer = "no"
        else:
            answer = "yes"
    if answer == None:
        answer = ' '.join(rec_check_for_pos(story["story_par"][0],["NP"]))
    
    answer = re.sub(r'[.,?!]', '', answer)  # keep colons for time answers
    return answer


#############################################################
###     Dont change the code below here
#############################################################

class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()

def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
