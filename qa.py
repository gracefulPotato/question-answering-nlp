
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import re, nltk
from nltk.corpus import stopwords
import itertools
from nltk.stem.wordnet import WordNetLemmatizer

should_normalize = True


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
            verb_index = node-2
            break
    #look for auxes and put them right before verb
    for node in dep_q.nodes:
        if dep_q.nodes[node]["rel"] == "aux":
            auxiliary = keywords[node-1]
            keywords.remove(auxiliary)
            keywords.insert(verb_index,auxiliary)
            #increment the verb's index since it's pushed one later            
            verb_index = verb_index + 1
    return keywords

def rec_check_for_pos(par_q,posses):
    pos_instances = []
    for child in par_q:
        if child in par_q.leaves():
            return []
        elif any(pos in child.label() for pos in posses):
            pos_instances = pos_instances + child.leaves()
        else: 
            pos_instances = pos_instances + rec_check_for_pos(child,posses)
    return pos_instances

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
    print(len(keyword_combos))
    all_keywords = "(?:"+re.sub("\?",""," ".join(keywords[1:]))+")"
    for keywords in keyword_combos:
        keyword = " ".join(keywords[1:])
        keyword = re.sub("\?","",keyword)
        all_keywords = all_keywords+"|(?:"+keyword+"[^\.]*)"
    return all_keywords


def get_keyword(question,dep_q):
    keywords = question.split()
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
        keyword = get_noun(question,dep_q)
        
    return keyword


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


# Match the question with the sentence with the most similar words
def question_answer_similarity(question_text, story):
    question_words = nltk.word_tokenize(question_text)
    text_sentences = nltk.sent_tokenize(story["text"])
    text_freq = {}
    for sentence in text_sentences:
        text_words = nltk.word_tokenize(sentence)
        text_freq[sentence] = 0
        for word in question_words:
            if word in text_words:
                text_freq[sentence] += 1

    return max(text_freq, key=text_freq.get)


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
        print("normalizing WHO: "+question_text)
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
    text_q = question["text"]
    dep_q = question["dep"]
    par_q = question["par"]

    print("\n")
    print(question["qid"])
    
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

    # for easy questions, match with regex
    if question_difficulty == "Easy":

        # Get question type (who, what, where, when, or why)
        question_type_who   = re.match(r'Who (.*)',   question_text)
        question_type_what  = re.match(r'What (.*)',  question_text)
        question_type_where = re.match(r'Where (.*)', question_text)
        question_type_when  = re.match(r'When (.*)',  question_text)
        question_type_why   = re.match(r'Why (.*)',   question_text)

        if question_type_who:
            print(question_type_who.group())
            keyword = get_keyword(question_type_who.group(),dep_q)
        if question_type_what:
            print(question_type_what.group())
        if question_type_where:
            print(question_type_where.group())
            keyword = get_keyword(question_type_where.group(),dep_q)
        if question_type_when:
            print(question_type_when.group())
        if question_type_why:
            print(question_type_why.group())
            keyword = get_noun(question_type_why.group(), dep_q)

        global should_normalize
        should_normalize = True

        lmtzr = WordNetLemmatizer()
        story_words = nltk.word_tokenize(story["text"].lower())
        lemmad_words = []
        for word in story_words:
            lemmad_words.append(lmtzr.lemmatize(word))
        print(keyword)
        matches = re.findall(("[^\.]*"+keyword+"[^\.]*").lower(),story["text"].lower())
        print(matches)
        if len(matches) != 0:
            if question_type_why:
                for match in matches:
                    sentence = re.findall("to|because", match)
                    if sentence:
                        return match
                    else:
                        continue
            return matches[0]
        
        else:
            answer = question_answer_similarity(question_text, story)
            

    elif question_difficulty == "Medium":
        print(question_difficulty)
        normalize_question(question_text,dep_q,par_q)
        answer = question_answer_similarity(question_text, story)

    else:
        print(question_difficulty)
        answer = question_answer_similarity(question_text, story)

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
