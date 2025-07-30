from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
import os

OPEN_AI_TOKEN = os.getenv("OPENAI_API_KEY")
print(f"first few chars of OPEN_AI_TOKEN: {OPEN_AI_TOKEN[:5]}...")
from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=OPEN_AI_TOKEN
)

class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.

    # Category of the email (inquiry, complaint, etc.)
    email_category: Optional[str]

    # Reason why the email was marked as spam
    spam_reason: Optional[str]

    # Analysis and decisions
    is_spam: Optional[bool]

    # Response generation
    email_draft: Optional[str]

    # Processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]

    # Here we might do some initial preprocessing
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")

    # No state changes needed here
    return {}

def classify_email(state: EmailState):
    email = state["email"]

    prompt = f"""
As Alfred the butler of Mr wayne and it's SECRET identity Batman, analyze this email and determine if it is spam or legitimate and should be brought to Mr wayne's attention.

Email:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

First, determine if this email is spam.
answer with SPAM or HAM if it's legitimate. Only return the answer
Answer :
    """
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    response_text = response.content.lower()
    print(response_text)
    is_spam = "spam" in response_text and "not spam" not in response_text

    # Extract the spam reason if available
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")

    # We're done processing this email
    return {}


def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"

    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    # Return state updates
    return {
        "email_draft": response.content,
        "messages": new_messages
    }


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state["email_draft"])
    print("=" * 50 + "\n")

    # We're done processing this email
    return {}
#💡Note: This routing function is called by LangGraph to determine which edge to follow after the classification node. The return value must match one of the keys in our conditional edges mapping.


def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"

if __name__ == "__main__":
    email_graph = StateGraph(EmailState)
    # Add nodes
    email_graph.add_node("read_email", read_email)
    email_graph.add_node("classify_email", classify_email)
    email_graph.add_node("handle_spam", handle_spam)
    email_graph.add_node("draft_response", draft_response)
    email_graph.add_node("notify_mr_hugg", notify_mr_hugg)
    # Start the edges
    email_graph.add_edge(START, "read_email")
    # Add edges - defining the flow
    email_graph.add_edge("read_email", "classify_email")

    # Add conditional branching from classify_email
    email_graph.add_conditional_edges(
        "classify_email",
        route_email,
        {
            "spam": "handle_spam",
            "legitimate": "draft_response"
        }
    )

    # Add the final edges
    email_graph.add_edge("handle_spam", END)
    email_graph.add_edge("draft_response", "notify_mr_hugg")
    email_graph.add_edge("notify_mr_hugg", END)

    # Compile the graph
    compiled_graph = email_graph.compile()
    # Example legitimate email
    legitimate_email = {
        "sender": "john.smith@example.com",
        "subject": "Question about your services",
        "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
    }

    # Example spam email
    spam_email = {
        "sender": "winner@lottery-intl.com",
        "subject": "YOU HAVE WON $5,000,000!!!",
        "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
    }

    # Process the legitimate email
    print("\nProcessing legitimate email...")
    legitimate_result = compiled_graph.invoke({
        "email": legitimate_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": []
    })

    # Process the spam email
    print("\nProcessing spam email...")
    spam_result = compiled_graph.invoke({
        "email": spam_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": []
    })

