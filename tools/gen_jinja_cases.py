"""Generates tests/data/jinja_cases.json: templates, inputs and the output real
Jinja produces for them, with the settings transformers uses for chat
templates (trim_blocks, lstrip_blocks, loop controls, raise_exception).

    python tools/gen_jinja_cases.py > tests/data/jinja_cases.json

Needs jinja2 (installed with transformers).
"""

import json
import sys
from datetime import datetime

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

CONVERSATION = [
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "  What is 2+2?  "},
]
WITH_SYSTEM = [{"role": "system", "content": "Be brief."}] + CONVERSATION
THINKING = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nHello!"},
    {"role": "user", "content": "And?"},
    {"role": "assistant", "content": "<think>\nagain\n</think>\n\nMore."},
]
NOT_ALTERNATING = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]

TINYLLAMA = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
SMOLLM2 = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
LLAMA3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
LLAMA2 = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
MISTRAL = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

# Mistral 7B Instruct v0.3's, with tool calling (unused here, but it must parse).
MISTRAL_V3 = '{%- if messages[0]["role"] == "system" %}\n    {%- set system_message = messages[0]["content"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}\n            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message["role"] == "user" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- "[AVAILABLE_TOOLS] [" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- \'{"type": "function", "function": {\' }}\n                {%- for key, val in tool.items() if key != "return" %}\n                    {%- if val is string %}\n                        {{- \'"\' + key + \'": "\' + val + \'"\' }}\n                    {%- else %}\n                        {{- \'"\' + key + \'": \' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- ", " }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- "}}" }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- else %}\n                    {{- "]" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- "[/AVAILABLE_TOOLS]" }}\n            {%- endif %}\n        {%- if loop.last and system_message is defined %}\n            {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "[/INST]" }}\n        {%- else %}\n            {{- "[INST] " + message["content"] + "[/INST]" }}\n        {%- endif %}\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\n        {{- "[TOOL_CALLS] [" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n            {%- endif %}\n            {{- \', "id": "\' + tool_call.id + \'"}\' }}\n            {%- if not loop.last %}\n                {{- ", " }}\n            {%- else %}\n                {{- "]" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message["role"] == "assistant" %}\n        {{- " " + message["content"]|trim + eos_token}}\n    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- \'[TOOL_RESULTS] {"content": \' + content|string + ", " }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n        {%- endif %}\n        {{- \'"call_id": "\' + message.tool_call_id + \'"}[/TOOL_RESULTS]\' }}\n    {%- else %}\n        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\n    {%- endif %}\n{%- endfor %}\n'

# Qwen3's, which reasons in <think> blocks and walks the messages backwards.
QWEN3 = '{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0].role == \'system\' %}\n        {{- messages[0].content + \'\\n\\n\' }}\n    {%- endif %}\n    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0].role == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith(\'<tool_response>\') and message.content.endswith(\'</tool_response>\')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = \'\' %}\n    {%- endif %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {%- set reasoning_content = \'\' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if \'</think>\' in content %}\n                {%- set reasoning_content = content.split(\'</think>\')[0].rstrip(\'\\n\').split(\'<think>\')[-1].lstrip(\'\\n\') %}\n                {%- set content = content.split(\'</think>\')[-1].lstrip(\'\\n\') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n            {%- else %}\n                {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- \'\\n\' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- \'<tool_call>\\n{"name": "\' }}\n                {{- tool_call.name }}\n                {{- \'", "arguments": \' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- \'}\\n</tool_call>\' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- \'<think>\\n\\n</think>\\n\\n\' }}\n    {%- endif %}\n{%- endif %}'

CHAT = {"bos_token": "<s>", "eos_token": "</s>", "add_generation_prompt": True}

# (name, template, context) - context values are plain JSON.
CASES = [
    # namespace(): an object whose attributes a set inside a loop can change.
    ("namespace", "{% set ns = namespace(total=0, seen=false) %}{% for x in [1, 2, 3] %}"
     "{% set ns.total = ns.total + x %}{% if x == 2 %}{% set ns.seen = true %}{% endif %}"
     "{% endfor %}{{ ns.total }} {{ ns.seen }}", {}),
    ("built-in functions are defined", "{{ strftime_now is defined }} {{ raise_exception is "
     "defined }} {{ range is defined }} {{ namespace is defined }} {{ nothing is defined }}", {}),
    ("namespace attribute added", "{% set ns = namespace() %}{% set ns.a = 'x' %}{{ ns.a }}", {}),
    ("attribute of a non-namespace", "{% set d = {'a': 1} %}{% set d.a = 2 %}{{ d.a }}", {}),
    # selectattr, and loops over the items passing a condition.
    ("selectattr equalto", "{{ msgs | selectattr('role', 'equalto', 'user') | list | length }} "
     "{{ (msgs | selectattr('role', 'equalto', 'user') | list)[-1].content }}",
     {"msgs": CONVERSATION}),
    ("selectattr truthy", "{% for i in items | selectattr('on') %}{{ i.n }}{% endfor %}",
     {"items": [{"n": 1, "on": True}, {"n": 2, "on": False}, {"n": 3}]}),
    ("for with a condition", "{% for x in [1, 2, 3, 4] if x % 2 == 0 %}{{ loop.index }}:{{ x }}"
     "{% if not loop.last %},{% endif %}{% endfor %}", {}),
    ("for over items with a condition", "{% for k, v in d.items() if k != 'b' %}{{ k }}={{ v }};"
     "{% endfor %}", {"d": {"a": 1, "b": 2, "c": 3}}),
    # Text, output and whitespace control.
    ("text", "plain text", {}),
    ("output", "a{{ x }}b", {"x": "X"}),
    ("undefined prints nothing", "[{{ missing }}]", {}),
    ("none and bools", "{{ none }} {{ true }} {{ false }}", {}),
    ("numbers", "{{ 1 + 2 }} {{ 7 // 2 }} {{ 7 % 3 }} {{ 2 * 3 - 1 }} {{ -4 }} {{ 1.5 + 1 }}", {}),
    ("string escapes", "{{ 'a\\nb\\t\"c\"' }}|{{ \"it's\" }}", {}),
    ("comment", "a{# gone #}b", {}),
    ("trim_blocks", "{% if true %}\nx\n{% endif %}\ny", {}),
    ("lstrip_blocks", "a\n    {% if true %}\n    b\n    {% endif %}\nc", {}),
    ("minus trims", "a   {{- 'b' -}}   c  {%- if true -%}  d  {%- endif %}", {}),
    ("output keeps newline", "{{ 'a' }}\n{{ 'b' }}\n", {}),
    # Statements.
    ("if elif else", "{% for n in [1, 2, 3] %}{% if n == 1 %}one{% elif n == 2 %}two{% else %}many{% endif %},{% endfor %}", {}),
    ("for loop variables", "{% for x in items %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}{{ loop.length }};{% endfor %}", {"items": ["a", "b"]}),
    ("for else", "{% for x in [] %}x{% else %}empty{% endfor %}", {}),
    ("break continue", "{% for n in [1, 2, 3, 4] %}{% if n == 2 %}{% continue %}{% endif %}{% if n == 4 %}{% break %}{% endif %}{{ n }}{% endfor %}", {}),
    ("set and scope", "{% set a = 1 %}{% for x in [1] %}{% set a = 2 %}{{ a }}{% endfor %}{{ a }}{% if true %}{% set b = 3 %}{% endif %}{{ b }}", {}),
    ("dict iteration", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}", {"d": {"x": 1, "y": "z"}}),
    # Expressions.
    ("comparisons", "{{ 1 < 2 }}{{ 2 <= 1 }}{{ 'a' == 'a' }}{{ 'a' != 'a' }}{{ 3 > 2 }}{{ 1 >= 1 }}", {}),
    ("logic", "{{ true and false }}{{ true or false }}{{ not true }}{{ not (1 == 2) }}", {}),
    ("in", "{{ 'b' in 'abc' }}{{ 2 in [1, 2] }}{{ 'k' in d }}{{ 'q' not in d }}", {"d": {"k": 1}}),
    ("tests", "{{ x is defined }}{{ y is defined }}{{ y is not defined }}{{ n is none }}{{ s is string }}", {"x": 1, "n": None, "s": "s"}),
    ("conditional expression", "{{ 'yes' if flag else 'no' }}{{ 'a' if false }}", {"flag": True}),
    ("concat operator", "{{ 'n=' ~ 3 ~ true }}", {}),
    ("subscript and slices", "{{ xs[0] }}{{ xs[-1] }}{{ xs[1:] }}{{ xs[:2] }}{{ 'hello'[1:3] }}{{ d['k'] }}{{ d.k }}", {"xs": [1, 2, 3], "d": {"k": "v"}}),
    ("slice steps", "{{ xs[::-1] }}{{ xs[::2] }}{{ xs[1::2] }}{{ xs[-1:0:-1] }}{{ xs[:-2:-1] }}{{ 'abc'[::-1] }}", {"xs": [1, 2, 3, 4, 5]}),
    ("list and dict literals", "{{ [1, 'a'] }}{{ {'k': 1}['k'] }}", {}),
    ("filters", "{{ '  x  ' | trim }}|{{ xs | length }}|{{ 'Ab' | upper }}{{ 'Ab' | lower }}|{{ xs | join('-') }}|{{ xs | first }}{{ xs | last }}|{{ missing | default('d') }}|{{ 5 | string }}", {"xs": [1, 2, 3]}),
    ("filter binds tighter than plus", "{{ 'a' + ' b ' | trim + 'c' }}", {}),
    ("string methods", "{{ ' x '.strip() }}{{ 'abc'.startswith('ab') }}{{ 'abc'.endswith('x') }}{{ 'ab'.upper() }}{{ 'hello world'.title() }}{{ ' x '.lstrip() }}|{{ ' x '.rstrip() }}", {}),
    ("strip characters", "[{{ '\\n\\nx\\n'.strip('\\n') }}][{{ 'xxay'.lstrip('x') }}][{{ 'ayyx'.rstrip('xy') }}][{{ ' a '.strip(none) }}]", {}),
    ("split", "{{ 'a<b>c<b>'.split('<b>') }}{{ ' a  b '.split() }}{{ 'x'.split(',') }}{{ 'r</think>a'.split('</think>')[-1] }}", {}),
    ("dict get", "{{ d.get('k') }}{{ d.get('q', 'dflt') }}", {"d": {"k": "v"}}),
    ("tojson", "{{ {'a': [1, 'x', none, true]} | tojson }}", {}),
    ("tojson indented", "{{ {'b': [1, {'c': 'x'}], 'a': {}, 'e': []} | tojson(indent=4) }}", {}),
    ("range", "{% for i in range(3) %}{{ i }}{% endfor %}", {}),
    ("range with a step", "{{ range(1, 8, 3) | list }}{{ range(3, -1, -1) | list }}{{ range(0, 3, -1) | list }}", {}),
    # Real chat templates.
    ("tinyllama chat", TINYLLAMA, {**CHAT, "messages": CONVERSATION}),
    ("tinyllama chat with system", TINYLLAMA, {**CHAT, "messages": WITH_SYSTEM}),
    ("smollm2 chat", SMOLLM2, {**CHAT, "messages": CONVERSATION}),
    ("smollm2 chat with system", SMOLLM2, {**CHAT, "messages": WITH_SYSTEM, "add_generation_prompt": False}),
    ("llama3 chat", LLAMA3, {**CHAT, "bos_token": "<|begin_of_text|>", "messages": WITH_SYSTEM}),
    ("llama2 chat", LLAMA2, {**CHAT, "messages": WITH_SYSTEM}),
    ("llama2 chat roles must alternate", LLAMA2, {**CHAT, "messages": NOT_ALTERNATING}),
    ("mistral chat", MISTRAL, {**CHAT, "messages": CONVERSATION}),
    ("mistral rejects system", MISTRAL, {**CHAT, "messages": WITH_SYSTEM}),
    ("qwen3 chat", QWEN3, {**CHAT, "messages": WITH_SYSTEM}),
    ("qwen3 chat drops earlier reasoning", QWEN3, {**CHAT, "messages": THINKING}),
    ("qwen3 chat without thinking", QWEN3, {**CHAT, "messages": CONVERSATION, "enable_thinking": False}),
    ("mistral v0.3 chat", MISTRAL_V3, {**CHAT, "messages": CONVERSATION}),
    ("mistral v0.3 chat with system", MISTRAL_V3, {**CHAT, "messages": WITH_SYSTEM}),
    ("mistral v0.3 roles must alternate", MISTRAL_V3, {**CHAT, "messages": NOT_ALTERNATING}),
]


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def main():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    # As transformers: LLaMA 3's templates print today's date with it.
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    # As transformers' own tojson, which keeps keys in order.
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    out = []
    for name, template, context in CASES:
        case = {"name": name, "template": template, "context": context}
        try:
            case["output"] = env.from_string(template).render(**context)
        except jinja2.exceptions.TemplateError as e:
            case["error"] = str(e)
        out.append(case)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
