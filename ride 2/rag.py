import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import openai

# Step 1: Load the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Step 2: Encode the data
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
patterns = []
responses = []

for intent in data["intents"]:
    patterns.extend(intent["patterns"])
    responses.extend(intent["responses"])

# Step 3: Set up the retrieval system
pattern_embeddings = model.encode(patterns)
d = pattern_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(pattern_embeddings)

# Step 4: Set up the language model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
llm = GPT2LMHeadModel.from_pretrained('gpt2')
openai.api_key = "your_openai_api_key_here"  # Replace with your OpenAI API key

# Step 5: Define a function to integrate retrieval, generation, and filtering
def generate_response(user_query):
    # Encode the user's query
    query_embedding = model.encode([user_query])
    
    # Perform vector search
    _, indices = index.search(query_embedding, k=1)
    # Get the most relevant context
    context_index = indices[0][0]
    context = responses[context_index]
    
    # Create a prompt with the user's query and context
    prompt = f"User: {user_query}\nTherapist: {context}\nTherapist: "
    
    # Generate a response using the language model
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    output = llm.generate(input_ids, max_length=100, num_return_sequences=1)
    
    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Filter the response using ChatGPT
    filtered_response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=response,
        temperature=0.6,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    ).choices[0].text.strip()
    
    # Return the filtered response
    return filtered_response

# Chat loop
def chat_with_ai_therapist():
    print("AI Therapist: Hello! How can I help you today?")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("AI Therapist: Goodbye! Take care.")
            break
        
        # Generate AI therapist response
        ai_response = generate_response(user_query)
        print(f"AI Therapist: {ai_response}")

# Start the chat
chat_with_ai_therapist()
