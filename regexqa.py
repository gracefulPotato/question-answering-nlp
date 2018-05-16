
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import re


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
        id  --  The id of the question.


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
    
    keyword = ""
    for node in dep_q.nodes:
        if dep_q.nodes[node]["rel"]=="nsubj":
            print(dep_q.nodes[node]["lemma"])
            keyword = dep_q.nodes[node]["lemma"].lower()
    numwords = len(text_q.split(" "))
    if keyword=="":
        print("try: "+str(dep_q.nodes[numwords]["lemma"]))
        keyword = dep_q.nodes[numwords]["lemma"]
    matches = re.findall(keyword+"[^\.]*",story["text"])
    print(matches)
    answer = "whatever you think the answer is"

    if len(matches)!=0:
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
