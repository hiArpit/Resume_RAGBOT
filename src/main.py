from rag_chain import build_rag_chain
# import the build_rag_chain function from your rag_chain.py module
# build_rag_chain() builds and returns a callable rag_ask() that runs the full RAG flow: retrieve relevant chunks -> build prompt -> call Gemini -> returns the answer

if __name__ == "__main__":
    # Will run only if this file runs directly
    ask = build_rag_chain()
    print("Eldoria RAG Chatbot Loaded! Ask anything (type 'exit' to quit)\n")

    try:
        while True:
            q = input("You: ")
            # Reading one line of input from the user and stores it in variable 'q'
            if q.lower() in ["exit", "quit"]:
                break
            # If the user enters exit or quit(case-insensitive), the loop breaks and the program finishes

            answer = ask(q)
            # Calling the ask() with the user query "q"
            # This performs the retrieval and a model call
            print("\nEldoriaBot:", answer, "\n")
            # Printing the returned answer
    except KeyboardInterrupt:
        print("\nBye!")