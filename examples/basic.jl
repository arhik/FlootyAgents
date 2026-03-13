using FlootyAgents

# Create a simple agent
agent = create_agent("MyAgent", :ollama; model="nemotron-3-nano:30b-cloud")
response = chat(agent, "Hello! What can you help me with?")
if !(response isa FlootyStreams.OllamaError)
    println("Assistant: ", FlootyAbstract.content(response.message))
end

result = chat_with_tools(agent, "What directory am I in? Use the pwd tool.")
result = chat_with_tools(agent, "Create a file at /tmp/agent_test.txt with content 'Hello from FlootyAgents'")
result = chat_with_tools(agent, "Read the file /tmp/agent_test.txt")
