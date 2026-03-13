using FlootyAgents
using FlootyAbstract: role, content, provider, context, add_message!, create_message
using FlootyContextManager: SimpleStack, get_messages

parent = create_agent("TaskOrchestrator")
add_message!(context(parent), create_message(provider(parent), "user",
    "Please analyze the current directory and report what files exist."))

analyzer = create_subagent(parent, "FileAnalyzer",
    SimpleStack(max_messages=5),
    "file listing task")

# Verify context was copied from parent
parent_msg_count = length(get_messages(context(parent)))
analyzer_msg_count = length(get_messages(context(analyzer)))

print_agent_tree(parent, maxdepth=2)

writer = create_subagent(parent, "ReportWriter",
    SimpleStack(max_messages=3),
    "report generation")
