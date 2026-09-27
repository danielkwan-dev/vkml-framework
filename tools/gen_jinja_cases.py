"""Generates tests/data/jinja_cases.json: templates, inputs and the output real
Jinja produces for them, with the settings transformers uses for chat
templates (trim_blocks, lstrip_blocks, loop controls, raise_exception).

    python tools/gen_jinja_cases.py > tests/data/jinja_cases.json

Needs jinja2 (installed with transformers).
"""

import json
import sys

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

CONVERSATION = [
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "content": "Hello! How can I help?"},
    {"role": "user", "content": "  What is 2+2?  "},
]
WITH_SYSTEM = [{"role": "system", "content": "Be brief."}] + CONVERSATION
NOT_ALTERNATING = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]

TINYLLAMA = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
SMOLLM2 = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
LLAMA3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
LLAMA2 = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
MISTRAL = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

CHAT = {"bos_token": "<s>", "eos_token": "</s>", "add_generation_prompt": True}

# (name, template, context) - context values are plain JSON.
CASES = [
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
    ("list and dict literals", "{{ [1, 'a'] }}{{ {'k': 1}['k'] }}", {}),
    ("filters", "{{ '  x  ' | trim }}|{{ xs | length }}|{{ 'Ab' | upper }}{{ 'Ab' | lower }}|{{ xs | join('-') }}|{{ xs | first }}{{ xs | last }}|{{ missing | default('d') }}|{{ 5 | string }}", {"xs": [1, 2, 3]}),
    ("filter binds tighter than plus", "{{ 'a' + ' b ' | trim + 'c' }}", {}),
    ("string methods", "{{ ' x '.strip() }}{{ 'abc'.startswith('ab') }}{{ 'abc'.endswith('x') }}{{ 'ab'.upper() }}{{ 'hello world'.title() }}{{ ' x '.lstrip() }}|{{ ' x '.rstrip() }}", {}),
    ("dict get", "{{ d.get('k') }}{{ d.get('q', 'dflt') }}", {"d": {"k": "v"}}),
    ("tojson", "{{ {'a': [1, 'x', none, true]} | tojson }}", {}),
    ("range", "{% for i in range(3) %}{{ i }}{% endfor %}", {}),
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
]


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def main():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
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
