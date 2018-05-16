
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import re ###


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
    ###     Your Code Goes Here         ###
    
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
        if question_type_what:
            print(question_type_what.group())
        if question_type_where:
            print(question_type_where.group())
        if question_type_when:
            print(question_type_when.group())
        if question_type_why:
            print(question_type_why.group())
        


    answer = "whatever you think the answer is"

    ###     End of Your Code         ###
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
