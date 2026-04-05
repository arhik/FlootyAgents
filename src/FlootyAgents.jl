module FlootyAgents

using FlootyAbstract
using FlootyAbstract: AbstractAgent, AbstractLLMProvider, AbstractContextStore, AbstractTool,
                      AbstractMessage, AbstractToolCall, AbstractFlootyError,
                      tool_name, call_name, call_arguments,
                      execute_tool, create_message, create_provider_instance, content, role,
                      response_message, response_has_tool_calls
import FlootyAbstract: create_agent, get_messages
using FlootyBase: Tool
using FlootyStreams: OllamaProvider, OllamaMessage, OllamaTool
using FlootyContextManager: ContextStore, add_message!, clear_history!, AbstractContextAlgorithm, SimpleStack, InfStack, get_context
using AbstractTrees
using UUIDs
using Dates
using OrderedCollections: OrderedDict

export Agent, reset!, add_child!, is_root, root, siblings, descendants, print_agent_tree
export add_tool!, remove_tool!
export create_subagent, select_context, launch_task, wait_for_results, launch_and_wait
export SimilaritySearch, KeywordMatch

include("agent.jl")

# ═══════════════════════════════════════════════════════════════════════
# StructTypes Configuration for Persistence
# ═══════════════════════════════════════════════════════════════════════

using StructTypes

# Agent - exclude runtime-only fields
StructTypes.StructType(::Type{Agent}) = StructTypes.Mutable()
StructTypes.excludes(::Type{Agent}) = (:channels, :active_tasks)

# Abstract type discrimination for provider
StructTypes.StructType(::Type{AbstractLLMProvider}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{AbstractLLMProvider}) = :_type
StructTypes.subtypes(::Type{AbstractLLMProvider}) = (
    OllamaProvider=OllamaProvider,
)

# Abstract type discrimination for context store
StructTypes.StructType(::Type{AbstractContextStore}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{AbstractContextStore}) = :_type
StructTypes.subtypes(::Type{AbstractContextStore}) = (
    ContextStore=ContextStore,
)

# Abstract type discrimination for tools
# FlootyAgents registers both provider-specific and generic tools
# since it can see both OllamaTool (from FlootyStreams) and Tool (from FlootyBase)
StructTypes.StructType(::Type{AbstractTool}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{AbstractTool}) = :_type
StructTypes.subtypes(::Type{AbstractTool}) = (
    OllamaTool=OllamaTool,  # Provider-specific tool from FlootyStreams
    Tool=Tool,               # Generic tool from FlootyBase
)

# Abstract type discrimination for Agent (used by Session)
StructTypes.StructType(::Type{AbstractAgent}) = StructTypes.AbstractType()
StructTypes.subtypekey(::Type{AbstractAgent}) = :_type
StructTypes.subtypes(::Type{AbstractAgent}) = (
    Agent=Agent,
)

# Note: Agent uses StructTypes.Mutable() with Agent() default constructor
# Excluded fields (channels, active_tasks) are initialized in Agent()
# Other fields are populated by StructTypes during deserialization

function FlootyAbstract.chat(agent::Agent, message::String;
                              stream::Bool=false,
                              auto_tools::Bool=true)
    msg = create_message(provider(agent), "user", message)
    add_message!(context(agent), msg)
    _do_chat(agent, stream, auto_tools)
end

# Note: _do_chat may return either a ChatResponse or the last ContextMessage
# Callers should handle both cases appropriately

# New: Chat with attachments
function FlootyAbstract.chat(agent::Agent, message::String, attachments::Vector{<:AbstractAttachment};
                              stream::Bool=false,
                              auto_tools::Bool=true)
    # Create message with attachments using provider
    prov = provider(agent)
    msg = create_message(prov, "user", message, attachments)
    add_message!(context(agent), msg)
    _do_chat(agent, stream, auto_tools)
end

# Internal function to handle the actual chat logic
function _do_chat(agent::Agent, stream::Bool, auto_tools::Bool)

    open("/tmp/agent.log", "a") do f
        println(f, "[$(now())] Chat starting for agent: ", agent.name)
        println(f, "[$(now())] Messages: ", length(get_messages(context(agent))))
    end

    iteration = 0
    while iteration < 100  # Safety limit
        iteration += 1
        messages = get_messages(context(agent))
        
        open("/tmp/agent.log", "a") do f
            println(f, "[$(now())] Iteration ", iteration, ", calling chat_stream...")
        end
        
        response = FlootyAbstract.chat_stream(provider(agent), agent.model, messages,
                                               stream, nothing, nothing, tools(agent))
        
        open("/tmp/agent.log", "a") do f
            println(f, "[$(now())] Got response: ", typeof(response))
        end

        if response isa AbstractFlootyError
            open("/tmp/agent.log", "a") do f
                println(f, "[$(now())] Error: ", response.message)
            end
            return response
        end

        resp_msg = response_message(response)

        open("/tmp/agent.log", "a") do f
            println(f, "[$(now())] Response message: content='$(content(resp_msg))' tool_calls=$(tool_calls(resp_msg))")
        end

        msg_content = content(resp_msg)
        has_tool_calls = response_has_tool_calls(response)
        if (msg_content !== nothing && !isempty(msg_content)) || has_tool_calls
            add_message!(context(agent), resp_msg)
            open("/tmp/agent.log", "a") do f
                println(f, "[$(now())] Added response to context (has_content=$(msg_content !== nothing && !isempty(msg_content)), has_tool_calls=$has_tool_calls)")
            end
        else
            open("/tmp/agent.log", "a") do f
                println(f, "[$(now())] Skipped empty response message")
            end
        end

        if stream || !auto_tools || !response_has_tool_calls(response)
            open("/tmp/agent.log", "a") do f
                println(f, "[$(now())] Returning response")
            end
            return response
        end

        open("/tmp/agent.log", "a") do f
            println(f, "[$(now())] Executing tools...")
        end
        
        for tc in tool_calls(resp_msg)
            tool_name_str = call_name(tc)
            tool_args = call_arguments(tc)
            result = execute_tool(tool_name_str, tool_args)
            
            # Format tool content like a shell command
            args_str = join(["$k=$v" for (k, v) in tool_args], " ")
            output = result.success ? result.data : "Error: $(result.error)"
            tool_content = """\$ $tool_name_str $args_str
$output"""
            
            tool_msg = create_message(provider(agent), "tool", tool_content)
            add_message!(context(agent), tool_msg)
        end
    end
    
    open("/tmp/agent.log", "a") do f
        println(f, "[$(now())] Max iterations reached")
    end
    
    # Return last message
    msgs = get_messages(context(agent))
    if !isempty(msgs)
        return last(msgs)
    else
        return OllamaError("Max iterations", 0)
    end
end

end # module FlootyAgents
