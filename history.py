from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from color import color_text
####
#### Conversation & context related functions
####
conversation_history = []
def get_follow_up_context():
    """Function to retrieve the follow-up context (only the last Q&A chain)

    Returns:
        All previous context for questions that begin with '>' up to the
        first question that doesn't (original question).
        In "Q: <question> A: <answer>" form.
    """
    global conversation_history
    context = ""
    for entry in reversed(conversation_history):
        if entry['question'].startswith(">"):
            context = f"Q: {entry['question'][1:].strip()}\nA: {entry['answer']}\n" + context
        else:
            # Stop accumulating context when a non-follow-up question is found
            context = f"Q: {entry['question'][1:].strip()}\nA: {entry['answer']}\n" + context
            break
    return context

def query_with_follow_up(question, query_engine):
    """Combines prior context with current question as necessary before querying

    Args:
        question: The user query ('>' prefix will collect prior context)
        query_engine: llama-index supports multiple types of query engines
    Returns:
        LLM generated response to the question (and optional context)
    """

    global conversation_history

    if question.startswith(">"):
        print(color_text("Adding prior Q&A to context.", "blue"))
        context = get_follow_up_context()
        question_without_prefix = question[1:].strip()
        question_with_context = f"{context}\nQ: {question_without_prefix}"
    else:
        question_with_context = question

    response = query_engine.query(question_with_context)

    conversation_history.append({
        "question": question,
        "answer": response.response,
    })

    return response
