
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import re
from nltk.corpus import stopwords
import itertools

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
    for node in dep_q.nodes:
        if "VB" in dep_q.nodes[node]["tag"]:
            verb_index = node
    #look for auxes and put them right before verb
    for node in dep_q.nodes:
        if dep_q.nodes[node]["rel"] == "aux":
            auxiliary = keywords[node-1]
            keywords.remove(auxiliary)
            keywords.insert(verb_index,auxiliary)
            #increment the verb's index since it's pushed one later            
            verb_index = verb_index + 1
    return keywords

def remove_stopwords(words):
    for i in range(len(words)):
        if re.sub("[^\w]","",words[i]) in stopwords.words('english'):
            words[i] = "[^\.]*"
    print("removed stopwords: "+str(words))
    return words

def permute_and_join(keywords):
    keyword_combos = list(itertools.permutations(keywords))
    print(keyword_combos)
    print(len(keyword_combos))
    all_keywords = "(?:"+re.sub("\?",""," ".join(keywords[1:]))+")"
    for keywords in keyword_combos:
        keyword = " ".join(keywords[1:])
        keyword = re.sub("\?","",keyword)
        all_keywords = all_keywords+"|(?:"+keyword+"[^\.]*)"
    return all_keywords

def get_keyword(question,dep_q):
    # print(question)
    # print("WHERE")
    print(' '.join(question.split()[3:]))
    
    keywords = question.split()
    #need to normalize verb (unless it's a stopword like be)
    keywords = normalize_verb(keywords,dep_q)
    #move auxiliaries like "might" to be before the verb
    keywords = move_auxiliaries(keywords,dep_q)
    keywords = remove_stopwords(keywords)
    #permute keywords to account for different orderings
    keyword = permute_and_join(keywords)
        
    return keyword

def get_noun(question,dep_q):  ###
    keywords = question.split()

    for node in dep_q.nodes:
        if node-1 in range(len(keywords)):
            if "NNP" or "NN" or "NNS" in dep_q.nodes[node]["tag"]:   # only gets last NN* in question
                keyword = keywords[node-1]
            else:
                keyword = "none"

    keyword = re.sub('[?,.!]', '', keyword)
    # print("keyword! ", end = "")
    # print(keyword)
    return keyword
                
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

    print("\n")
    print(question["qid"])
    print(text_q)
    print(story["text"])
    
    # print(dep_q)
    
    keyword = ""
    for node in dep_q.nodes:
        if dep_q.nodes[node]["rel"]=="nsubj":
            #print(dep_q.nodes[node]["lemma"])
            keyword = dep_q.nodes[node]["lemma"] #.lower()
    numwords = len(text_q.split(" "))
    if keyword=="":
        for node in dep_q.nodes:
            if dep_q.nodes[node]["rel"]=="nobj":
                keyword = dep_q.nodes[node]["lemma"] #.lower()
    if keyword=="":
        print("try: "+str(dep_q.nodes[numwords]["lemma"]))
        keyword = dep_q.nodes[numwords]["lemma"]

###############################################
    
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
            #question_type_who.group()[2:]
        if question_type_what:
            print(question_type_what.group())
            #keyword = get_keyword(question_type_what.group(),dep_q)
        if question_type_where:
            print(question_type_where.group())
            print("WHERE")
            print(' '.join(question_type_where.group().split()[3:]))
            keyword = get_keyword(question_type_where.group(),dep_q)
            print(keyword)
        if question_type_when:
            print(question_type_when.group())
            #keyword = get_keyword(question_type_when.group(),dep_q)
        if question_type_why:
            print(question_type_why.group())
            keyword = get_noun(question_type_why.group(), dep_q)
            #keyword = get_keyword(question_type_why.group(),dep_q)
            
        print("matching keyword: "+keyword)
        matches = re.findall(("[^\.]*"+keyword+"[^\.]*").lower(),story["text"].lower())
        print("matches: ", end="")
        print(matches)

                    
    answer = "whatever you think the answer is"

    if len(matches)!=0:
        if question_type_why:   ###
            for match in matches:
                print("MATCH")
                print(match)
                sentence = re.findall("to|because", match)
                if sentence:
                    return match
                else:
                    continue
        return matches[0]
    else:
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
