

"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit - Batch 2026

Goal: A simple rule-based chatbot that responds to predefined user
inputs using if-else / dictionary logic, running in a continuous loop.
"""

# ------------------------------------------------------------------
# PHASE 1: KNOWLEDGE BASE
# A dictionary (hash map) gives O(1) constant-time lookup instead of
# a long, unstable if-elif ladder (O(n)).
# ------------------------------------------------------------------
responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a bunch of if-else statements, but I'm doing great!",
    "what is your name": "I'm ChatBot-1, DecodeLabs' first rule-based intern project.",
    "what can you do": "Right now I can chat using simple predefined rules. No AI magic yet!",
    "help": "Try greeting me, asking my name, or just say 'bye' to leave.",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
}

# Exit commands that break the loop (the "Kill Command")
exit_commands = {"exit", "bye", "quit", "goodbye"}

# Fallback response used when no rule matches
DEFAULT_RESPONSE = "I do not understand. Try typing 'help' to see what I can do."


def get_response(user_input: str) -> str:
    """Look up a response for the sanitized user input, with fallback."""
    return responses.get(user_input, DEFAULT_RESPONSE)


def main():
    print("ChatBot: Hello! I'm your rule-based assistant. Type 'bye' to exit.")

    # PHASE 2: THE HEARTBEAT - continuous input loop
    while True:
        raw_input_text = input("You: ")

        # PHASE 1: SANITIZATION - normalize case and whitespace
        clean_input = raw_input_text.lower().strip()

        # KILL COMMAND - clean exit strategy
        if clean_input in exit_commands:
            print("ChatBot: Goodbye! Have a great day.")
            break

        # PROCESS - intent matching via dictionary lookup
        reply = get_response(clean_input)
        print(f"ChatBot: {reply}")


if __name__ == "__main__":
    main()
