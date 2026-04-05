using FlootyAbstract: response_message

mutable struct Agent <: AbstractAgent
    id::UUID
    name::String
    provider::Union{AbstractLLMProvider,Nothing}
    context::Union{AbstractContextStore,Nothing}
    tools::Vector{AbstractTool}
    model::String
    created_at::DateTime
    metadata::Dict{String,Any}
    children::Vector{Agent}
    parent::Union{Agent,Nothing}

    channels::Dict{UUID, Channel{AbstractMessage}}
    active_tasks::Dict{UUID, Task}

    # Empty constructor for StructTypes deserialization
    Agent() = new(UUIDs.uuid4(), "", nothing, nothing, AbstractTool[], "", now(), Dict{String,Any}(), Agent[], nothing, Dict{UUID,Channel{AbstractMessage}}(), Dict{UUID,Task}())

    function Agent(name::String, provider::AbstractLLMProvider;
                   model::String="nemotron-3-nano:30b-cloud",
                   tools::Vector{<:AbstractTool}=AbstractTool[],
                   metadata::Dict{String,Any}=Dict{String,Any}())
        new(UUIDs.uuid4(), name, provider, ContextStore(),
            tools, model, now(), metadata, Agent[], nothing,
            Dict{UUID,Channel{AbstractMessage}}(), Dict{UUID,Task}())
    end
    function Agent(name::String, provider::AbstractLLMProvider, context::ContextStore;
                   model::String="nemotron-3-nano:30b-cloud",
                   tools::Vector{<:AbstractTool}=AbstractTool[],
                   metadata::Dict{String,Any}=Dict{String,Any}())
        new(UUIDs.uuid4(), name, provider, context,
            tools, model, now(), metadata, Agent[], nothing,
            Dict{UUID,Channel{AbstractMessage}}(), Dict{UUID,Task}())
    end
end

FlootyAbstract.role(agent::Agent) = agent.name
FlootyAbstract.provider(agent::Agent) = agent.provider === nothing ? error("Agent has no provider") : agent.provider
FlootyAbstract.context(agent::Agent) = agent.context === nothing ? error("Agent has no context") : agent.context
FlootyAbstract.tools(agent::Agent) = agent.tools

function add_child!(parent::Agent, child::Agent)
    child.parent = parent
    push!(parent.children, child)
end

function create_agent(name::String, provider_id::Symbol=:ollama; kwargs...)
    provider = create_provider_instance(Val(provider_id))
    Agent(name, provider; kwargs...)
end

function add_tool!(agent::Agent, tool::AbstractTool)
    push!(agent.tools, tool)
    agent
end

function remove_tool!(agent::Agent, tool_name::String)
    filter!(t -> FlootyAbstract.tool_name(t) != tool_name, agent.tools)
    agent
end

function reset!(agent::Agent)
    clear_history!(agent.context)
    agent
end

function agent_summary(agent::Agent)
    msg_count = length(get_messages(agent.context))
    tool_names = join(map(tool_name, agent.tools), ", ")
    child_count = length(agent.children)
    parent_name = agent.parent === nothing ? "none" : agent.parent.name
    """
    Agent: $(agent.name)
    ID: $(agent.id)
    Model: $(agent.model)
    Messages: $(msg_count)
    Tools: $(tool_names)
    Parent: $(parent_name)
    Children: $(child_count)
    Created: $(agent.created_at)
    """
end

using AbstractTrees

AbstractTrees.children(agent::Agent) = agent.children
AbstractTrees.parent(agent::Agent) = agent.parent

function print_agent_tree(agent::Agent; maxdepth=5)
    print_tree(stdout, agent, maxdepth=maxdepth)
end

function is_root(agent::AbstractAgent)
    agent.parent === nothing
end

function root(agent::AbstractAgent)
    current = agent
    while current.parent !== nothing
        current = current.parent
    end
    current
end

function siblings(agent::AbstractAgent)
    agent.parent === nothing && return AbstractAgent[]
    filter(c -> c !== agent, agent.parent.children)
end

function descendants(agent::AbstractAgent)
    result = AbstractAgent[]
    for child in agent.children
        push!(result, child)
        append!(result, descendants(child))
    end
    result
end

# Context selection algorithms
struct SimilaritySearch <: AbstractContextAlgorithm
    top_k::Int
    threshold::Float64
end

struct KeywordMatch <: AbstractContextAlgorithm  
    keywords::Vector{String}
    max_results::Int
end

SimilaritySearch(;top_k::Int=5, threshold::Float64=0.7) = SimilaritySearch(top_k, threshold)
KeywordMatch(keywords::Vector{String}; max_results::Int=10) = KeywordMatch(keywords, max_results)

function select_context(store::ContextStore, 
                        algorithm::SimilaritySearch, 
                        query::String)
    get_context(store, SimpleStack(max_messages=algorithm.top_k))
end

function select_context(store::ContextStore,
                        algorithm::KeywordMatch,
                        _query::String)
    get_messages(store)
end

function select_context(store::ContextStore,
                        algorithm::SimpleStack,
                        _query::String)
    get_context(store, algorithm)
end

function select_context(store::ContextStore,
                        algorithm::InfStack,
                        _query::String)
    get_context(store, algorithm)
end

# Subagent coordination functions
function create_subagent(parent::Agent, name::String,
                        context_algorithm::AbstractContextAlgorithm, 
                        query::String;
                        initial_prompt::String="")
    
    selected_messages = select_context(parent.context, context_algorithm, query)
    
    sub_context = ContextStore()
    for msg in selected_messages
        add_message!(sub_context, msg)
    end
    
    if !isempty(initial_prompt)
        system_msg = create_message(parent.provider, "system", initial_prompt)
        add_message!(sub_context, system_msg)
    end
    
    subagent = Agent(name, parent.provider, sub_context;
                    model=parent.model,
                    tools=parent.tools)
    
    add_child!(parent, subagent)
    
    channel = Channel{AbstractMessage}(10)
    parent.channels[subagent.id] = channel
    
    subagent
end

function create_result_message(agent_id::UUID, status::Symbol, data::String)
    content = """[TaskResult from $agent_id]
Status: $status
Data: $data"""
    OllamaMessage(role="tool", content=content)
end

function launch_task(parent::Agent, child_id::UUID, task_description::String)
    child_idx = findfirst(c -> c.id == child_id, parent.children)
    child = parent.children[child_idx]
    channel = parent.channels[child_id]
    
    task_msg = create_message(parent.provider, "user", task_description)
    add_message!(child.context, task_msg)
    
    task = @async begin
        try
            response = chat(child, ""; auto_tools=true)
            
            if response isa OllamaError
                result_msg = create_result_message(child_id, :error, 
                    "OllamaError: $(response.message)")
            else
                result_content = content(response_message(response))
                result_msg = create_result_message(child_id, :success, result_content)
            end
            
            put!(channel, result_msg)
            
        catch e
            result_msg = create_result_message(child_id, :error, 
                "Exception: $(sprint(showerror, e))")
            put!(channel, result_msg)
        end
    end
    
    parent.active_tasks[child_id] = task
    task
end

function wait_for_results(parent::Agent; timeout::Float64=60.0)
    results = AbstractMessage[]
    start_time = time()
    
    for (child_id, task) in parent.active_tasks
        if time() - start_time > timeout
            @warn "Timeout waiting for child $child_id"
            continue
        end
        
        wait(task)
        
        channel = parent.channels[child_id]
        if isready(channel)
            result = take!(channel)
            push!(results, result)
            add_message!(parent.context, result)
        end
    end
    
    empty!(parent.active_tasks)
    
    results
end

function launch_and_wait(parent::Agent, child_id::UUID, task::String; timeout::Float64=60.0)
    launch_task(parent, child_id, task)
    results = wait_for_results(parent; timeout=timeout)
    length(results) == 1 ? results[1] : results
end
