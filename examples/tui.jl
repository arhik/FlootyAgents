# ═══════════════════════════════════════════════════════════════════════
# FlootyAgents TUI - Agent-based Chat Interface
#
# Type messages and press Enter to chat with current agent.
# Use Ctrl+C or Esc to quit.
# ═══════════════════════════════════════════════════════════════════════

using Tachikoma
using FlootyAgents
using FlootyAgents: Agent, create_agent, is_root, root, print_agent_tree
using FlootyAbstract
using FlootyContextManager: add_message!, get_messages

# Use FlootyAbstract interface functions
const context = FlootyAbstract.context
const provider = FlootyAbstract.provider
using CommonMark
using Dates
using JSON3

@tachikoma_app

# ═══════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════
const GREY_TEXT = Style(fg=Color256(245))
const GREY_DIM = Style(fg=Color256(239))
const GREY_BG = Style(bg=ColorRGB(40, 40, 40))
const USER_BG = Style(bg=ColorRGB(60, 60, 60))
const ASSISTANT_BG = Style(bg=ColorRGB(40, 40, 60))
const TOOL_BG = Style(bg=ColorRGB(40, 80, 40))
const SYSTEM_BG = Style(bg=ColorRGB(80, 60, 40))
const SEPARATOR = Style(fg=Color256(239))

# ═══════════════════════════════════════════════════════════════════════
# Message Widget
# ═══════════════════════════════════════════════════════════════════════
mutable struct MessageWidget
    message::AbstractMessage
    agent_name::String
    width::Int
    
    function MessageWidget(msg::AbstractMessage, agent_name::String, width::Int)
        new(msg, agent_name, width)
    end
end

function render_widget(buf::Buffer, widget::MessageWidget, x::Int, y::Int, max_y::Int)
    y >= max_y && return 0
    
    msg = widget.message
    msg_role = role(msg)
    msg_content = content(msg) === nothing ? "" : content(msg)
    
    # Style based on role
    bg_style = if msg_role == "user"
        Style(bg=ColorRGB(60, 60, 60), fg=ColorRGB(255, 255, 255))
    elseif msg_role == "assistant"
        Style(bg=ColorRGB(40, 40, 60), fg=ColorRGB(200, 200, 255))
    elseif msg_role == "tool"
        Style(bg=ColorRGB(40, 80, 40), fg=ColorRGB(200, 255, 200))
    else
        Style(bg=ColorRGB(80, 60, 40), fg=ColorRGB(255, 220, 180))
    end
    
    # Header line
    header = "$(widget.agent_name) [$(msg_role)]"
    set_string!(buf, x + 1, y, header, GREY_DIM)
    
    # Content lines (simple wrap)
    lines = _wrap_lines(msg_content, widget.width - 4)
    height = 1 + length(lines)
    
    for (i, line) in enumerate(lines)
        line_y = y + i
        line_y >= max_y && break
        
        # Fill background
        for cx in (x+1):(x+widget.width-2)
            set_char!(buf, cx, line_y, ' ', bg_style)
        end
        
        set_string!(buf, x + 2, line_y, line, bg_style)
    end
    
    height
end

function _wrap_lines(text::String, width::Int)
    words = split(text, r"\s+", keepempty=false)
    lines = String[]
    current = ""
    
    for word in words
        if isempty(current)
            current = word
        elseif length(current) + 1 + length(word) <= width
            current *= " " * word
        else
            push!(lines, current)
            current = word
        end
    end
    !isempty(current) && push!(lines, current)
    
    isempty(lines) && push!(lines, "")
    lines
end

# ═══════════════════════════════════════════════════════════════════════
# TUI Model
# ═══════════════════════════════════════════════════════════════════════
@kwdef mutable struct TuiModel <: Model
    quit::Bool = false
    tick::Int = 0
    agent::Agent = create_agent("User")
    input::TextInput = TextInput(text="", focused=true, tick=0)
    task_queue::TaskQueue = TaskQueue()
    is_streaming::Bool = false
    show_tree::Bool = false
    scroll_offset::Int = 0
    focus::Int = 2
end

should_quit(m::TuiModel) = m.quit
task_queue(m::TuiModel) = m.task_queue

# ═══════════════════════════════════════════════════════════════════════
# Event Handling
# ═══════════════════════════════════════════════════════════════════════
function update!(m::TuiModel, evt::Event)
    if evt isa KeyEvent
        _handle_key!(m, evt)
    end
end

function _handle_key!(m::TuiModel, evt::KeyEvent)
    # Quit
    if (evt.key == :char && evt.char == '\x03') || evt.key == :escape
        m.quit = true
        return
    end
    
    # Toggle tree view
    if evt.key == :f2
        m.show_tree = !m.show_tree
        return
    end
    
    # Scroll
    if evt.key == :page_up
        m.scroll_offset = max(0, m.scroll_offset - 5)
        return
    elseif evt.key == :page_down
        m.scroll_offset += 5
        return
    end
    
    # Input handling
    if m.focus == 2
        handled = handle_key!(m.input, evt)
        if !handled && (String(m.input.buffer) |> strip |> !isempty) &&
           (evt.key == :enter || (evt.key == :char && evt.char == '\n'))
            _send_message!(m)
        end
    end
end

function _send_message!(m::TuiModel)
    text = String(strip(String(m.input.buffer)))
    isempty(text) && return
    
    empty!(m.input.buffer)
    m.input.cursor = 0
    
    # Log to file for debugging (don't use print in TUI)
    open("/tmp/tui.log", "a") do f
        println(f, "[$(now())] Sending: $text")
    end
    
    # Launch chat - it will add the user message
    m.is_streaming = true
    
    # Use @async directly with Channel for result
    result_channel = Channel{Any}(1)
    @async begin
        open("/tmp/tui.log", "a") do f
            println(f, "[$(now())] Task starting...")
        end
        try
            open("/tmp/tui.log", "a") do f
                println(f, "[$(now())] About to call chat()...")
            end
            result = FlootyAbstract.chat(m.agent, text; stream=false, auto_tools=true)
            open("/tmp/tui.log", "a") do f
                println(f, "[$(now())] Result: $(typeof(result))")
            end
            put!(result_channel, result)
        catch e
            open("/tmp/tui.log", "a") do f
                println(f, "[$(now())] Caught error: $(sprint(showerror, e))")
            end
            put!(result_channel, e)
        end
    end
    
    # Store channel for polling in update loop
    # (simplified: just check channel in next tick)
    # For now, spawn a task that waits and adds to context
    @async begin
        result = take!(result_channel)
        m.is_streaming = false
        if result isa Exception
            add_message!(context(m.agent), create_message(provider(m.agent), "system", 
                "⚠️ Error: $(sprint(showerror, result))"))
        elseif result isa FlootyAbstract.AbstractFlootyError
            add_message!(context(m.agent), create_message(provider(m.agent), "system", 
                "⚠️ Chat error: $(result.message)"))
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════
function view(m::TuiModel, f::Frame)
    m.tick += 1
    buf = f.buffer
    
    # Layout
    tree_width = m.show_tree ? 30 : 0
    chat_width = f.area.width - tree_width
    
    tree_area = Rect(f.area.x, f.area.y, tree_width, f.area.height - 3)
    chat_area = Rect(f.area.x + tree_width, f.area.y, chat_width, f.area.height - 3)
    input_area = Rect(f.area.x + tree_width, f.area.y + f.area.height - 3, chat_width, 3)
    
    # Draw separator if tree visible
    if m.show_tree
        sep_x = f.area.x + tree_width
        for y in f.area.y:(f.area.y + f.area.height - 4)
            set_char!(buf, sep_x, y, '┃', SEPARATOR)
        end
    end
    
    # Render tree sidebar
    if m.show_tree
        _render_tree!(m, buf, tree_area)
    end
    
    # Render chat messages
    _render_chat!(m, buf, chat_area)
    
    # Render input
    _render_input!(m, buf, input_area)
    
    # Status bar
    status = "$(role(m.agent)) | Ctrl+C quit | F2 tree"
    m.is_streaming && (status = "⏳ $status")
    set_string!(buf, f.area.x, f.area.y + f.area.height - 1, 
                status, GREY_DIM)
end

function _render_tree!(m::TuiModel, buf::Buffer, area::Rect)
    # Show agent tree starting from root
    root_agent = is_root(m.agent) ? m.agent : root(m.agent)
    
    title_y = area.y
    set_string!(buf, area.x + 1, title_y, "Agent Tree:", GREY_TEXT)
    
    # Simple tree print
    y = title_y + 1
    indent = is_root(m.agent) ? "▸ " : "  ▸ "
    
    set_string!(buf, area.x + 1, y, indent * role(root_agent), 
                Style(fg=ColorRGB(100, 255, 100)))
    
    if !is_root(m.agent)
        y += 1
        set_string!(buf, area.x + 3, y, "▸ " * role(m.agent), 
                    Style(fg=ColorRGB(255, 255, 255)))
    end
    
    # Show children if any
    for child in m.agent.children
        y += 1
        y > bottom(area) && break
        set_string!(buf, area.x + 5, y, "└ " * role(child), GREY_DIM)
    end
end

function _render_chat!(m::TuiModel, buf::Buffer, area::Rect)
    messages = get_messages(context(m.agent))
    
    rendered_height = 0
    for msg in messages
        widget = MessageWidget(msg, role(m.agent), area.width)
        y = area.y - m.scroll_offset + rendered_height
        
        if y >= area.y && y < bottom(area) - 1
            h = render_widget(buf, widget, area.x, y, bottom(area) - 1)
            rendered_height += h + 1
        end
    end
end

function _render_input!(m::TuiModel, buf::Buffer, area::Rect)
    # Draw background
    for y in area.y:bottom(area)
        for x in area.x:(area.x + area.width - 1)
            set_char!(buf, x, y, ' ', GREY_BG)
        end
    end
    
    # Input prompt
    prompt = " > "
    set_string!(buf, area.x + 1, area.y + 1, prompt, GREY_TEXT)
    
    # Render text input
    inner_area = Rect(area.x + 4, area.y + 1, area.width - 5, 1)
    m.input.focused = true
    m.input.tick = m.tick
    render(m.input, inner_area, buf)
end

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
function flooty_tui(; agent_name::String="User", 
                      model::String="nemotron-3-nano:30b-cloud")
    model_tui = TuiModel(agent=create_agent(agent_name; model=model))
    app(model_tui; fps=30)
end

# Run: flooty_tui()
