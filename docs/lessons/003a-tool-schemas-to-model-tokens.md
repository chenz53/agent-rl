# Lesson 3A: From tool schemas to the model's actual input tokens

Read this chapter after the opening tool example in [lesson 3](003-oncology-tools-and-environment.md), before its runtime-state class. It answers a concrete question: **when our agent requests performance status, exactly which objects exist, what is returned, and what reaches the model?**

The earlier score-only printout was a diagnostic projection. It was not the tool response. This chapter displays the full response and tracks every serialization boundary so that `PS-001`, `PS-002`, their timestamps, and their scores remain identifiable.

All Python blocks execute sequentially in one fresh session in the existing vp checkout, with `PYTHONPATH=src .venv/bin/python`. Outputs are captured below each block. We load the cached Qwen3-0.6B **tokenizer**, the real verl parser, and the real domain tool. We do not load model weights, sample a model response, call a hosted API, or run the complete AgentLoop. A hand-authored assistant continuation is explicitly marked when used.

## 1. First separate six objects that are often all called a schema

| Object | Who defines it? | Purpose | Automatically visible to the model? |
| --- | --- | --- | --- |
| CSV/data contract | Domain ingestion code | Interpret rows, timestamps, versions, units | No |
| Tool input schema | Tool author | Describe allowed callable arguments | Only if passed through the model's input/template path |
| Backend implementation/configuration | Application and framework | Bind the name to executable logic and patient context | No |
| Tool result payload | Tool author | Return evidence, provenance, or an error | Only when inserted into model-visible context |
| Message/call envelope | Chosen runtime/API | Represent roles and associate requests with results | Depends on template rendering; not every metadata field is rendered |
| Chat template and token IDs | Model/tokenizer integration | Convert the conversation and tools into the sequence used by the model | This is the actual textual input representation in our example |

A JSON Schema declaration does not contain the Python implementation. A Python object in server memory does not automatically become an input token. A printed list of scores does not replace the returned records unless code explicitly performs that replacement.

The path we will inspect is:

```text
oncology_tools.yaml
  -> model-visible tool declaration + backend configuration
  -> template(messages, tools)
  -> input token IDs
  -> [a real deployment would generate assistant tokens here]
  -> parser -> tool name and arguments
  -> registry -> create/execute/release
  -> ToolResponse.text containing the complete domain result
  -> tool message -> rendered continuation context -> token IDs
```

## 2. Inspect the real configuration, not a shortened substitute

```python
import asyncio
import contextlib
import io
import json
from nanoastra.oncology import ROOT, TABLES, tool_view
from nanoastra.upstream import activate_verl
activate_verl()
from verl.tools.tool_registry import initialize_tools_from_config

print((ROOT / "configs/oncology_tools.yaml").read_text().rstrip())
with contextlib.redirect_stdout(io.StringIO()):
    tool = initialize_tools_from_config(str(ROOT / "configs/oncology_tools.yaml"))[0]
print("Instantiated class:", type(tool).__name__)
```

Output:

```text
tools:
  - class_name: nanoastra.verl_tool.ClinicalRecordTool
    config:
      type: native
      record_fields: [patients, encounters, diagnosis, labs, vitals, notes, pathology, staging, biomarkers, medication_orders, medication_administrations, procedures, performance_status]
    tool_schema:
      type: function
      function:
        name: get_clinical_record
        description: Read one clinical table for the server-selected patient and index time. Outcomes and future records are inaccessible.
        parameters:
          type: object
          properties:
            field:
              type: string
              description: Allowed clinical table. Empty rows mean no available records, not a confirmed negative finding.
              enum: [patients, encounters, diagnosis, labs, vitals, notes, pathology, staging, biomarkers, medication_orders, medication_administrations, procedures, performance_status]
          required: [field]
Instantiated class: ClinicalRecordTool
```

There are two different namespaces here. `class_name` tells Python which implementation to instantiate. The nested `function.name` is the callable name advertised to the model. These names need not be identical: one is an import path, the other is an action identifier.

`config.record_fields` enforces the backend allowlist. The schema's `enum` describes allowed arguments to the model. They currently agree, but maintaining that agreement is an engineering responsibility, not a property of YAML. No patient ID or prediction cutoff appears in the callable parameters, because the environment chooses those values.

The `type: native` setting belongs to this framework configuration. It is not part of the clinical payload or a universal tool protocol.

## 3. Identify exactly what the declaration tells the model

The following is the complete parsed model-visible declaration, not the complete backend configuration:

```python
schema = tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
print(json.dumps(schema, indent=2))
assert schema["function"]["parameters"]["properties"]["field"]["enum"] == list(TABLES)
print("Schema/backend field lists match:", True)
```

Output:

```text
{
  "type": "function",
  "function": {
    "name": "get_clinical_record",
    "description": "Read one clinical table for the server-selected patient and index time. Outcomes and future records are inaccessible.",
    "parameters": {
      "type": "object",
      "properties": {
        "field": {
          "type": "string",
          "description": "Allowed clinical table. Empty rows mean no available records, not a confirmed negative finding.",
          "enum": [
            "patients",
            "encounters",
            "diagnosis",
            "labs",
            "vitals",
            "notes",
            "pathology",
            "staging",
            "biomarkers",
            "medication_orders",
            "medication_administrations",
            "procedures",
            "performance_status"
          ]
        }
      },
      "required": [
        "field"
      ]
    }
  }
}
Schema/backend field lists match: True
```

- `name` identifies the callable action.
- `description` explains the intended use and information boundary in natural language.
- `parameters.type = object` requires an argument object.
- `properties.field.type = string` describes the field's type.
- `enum` lists the allowed table names.
- `required` specifies that the field must be present.

This is an input contract, not the eventual return value. Our current YAML does not declare an output JSON Schema. It also does not set `additionalProperties: false`; the adapter nevertheless rejects unexpected argument keys in its runtime check. Describing a restriction and enforcing it are different layers. This chapter inspects existing behavior rather than silently adding a strict-decoding guarantee.

A syntactically valid argument still needs semantic and authorization checks. `{"field":"mortality"}` is valid JSON, but it is an invalid action in this environment.

## 4. Keep the patient snapshot on the server side

```python
records = tool_view()
print("Server-side table count:", len(records))
print("Stored table value type:", type(records["performance_status"]).__name__)
print("Model-selectable argument names:", list(schema["function"]["parameters"]["properties"]))
print("Backend configuration advertised in schema:", "class_name" in schema)
```

Output:

```text
Server-side table count: 13
Stored table value type: str
Model-selectable argument names: ['field']
Backend configuration advertised in schema: False
```

The actual adapter's `create(create_kwargs={"record": records})` receives these records. The model does not generate `create_kwargs`. Giving the model an arbitrary patient selector or file path would define a different action space and require different access controls.

A table being in `records` establishes only that the runtime can look it up. It does not establish that the LLM has read it. We can check that distinction at the rendered input, not merely assume it from container names.

## 5. Render the real initial prompt with the cached Qwen template

```python
from pathlib import Path
from transformers import AutoTokenizer
model_path = Path("/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
messages = [
    {"role": "system", "content": "Use only baseline records. Cite record IDs and state uncertainty."},
    {"role": "user", "content": "Assess prognosis for SYN-ONC-001 at 2025-03-01T09:00:00Z. Read evidence as needed."},
]
rendered = tokenizer.apply_chat_template(messages, tools=[schema], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
print(rendered)
```

Output:

```text
<|im_start|>system
Use only baseline records. Cite record IDs and state uncertainty.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_clinical_record", "description": "Read one clinical table for the server-selected patient and index time. Outcomes and future records are inaccessible.", "parameters": {"type": "object", "properties": {"field": {"type": "string", "description": "Allowed clinical table. Empty rows mean no available records, not a confirmed negative finding.", "enum": ["patients", "encounters", "diagnosis", "labs", "vitals", "notes", "pathology", "staging", "biomarkers", "medication_orders", "medication_administrations", "procedures", "performance_status"]}}, "required": ["field"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Assess prognosis for SYN-ONC-001 at 2025-03-01T09:00:00Z. Read evidence as needed.<|im_end|>
<|im_start|>assistant
<think>

</think>
```

This is the full initial rendered string. The template places the tool declaration inside a `<tools>` section and tells the model how to format a request inside `<tool_call>` tags. The available tool name, field descriptions, and allowed values are now part of the context.

Notice what is absent: the CSV rows, backend class path, `PS-001`, and `PS-002`. The table name `performance_status` is visible because it is an allowed action argument; the contents of that table are not yet visible. Knowing that a tool exists is different from seeing its results.

The empty thinking delimiters are supplied by this particular template when `enable_thinking=False`. They are not a reasoning trace or tokens generated by a model during this demonstration. Template options and control markers are model-specific.

## 6. How does the model capture the schema?

The string above is tokenized. A transformer consumes token IDs (and applicable masks/positions), not a Python dictionary with built-in JSON validation semantics.

```python
input_ids = tokenizer.encode(rendered, add_special_tokens=False)
print("Initial token count:", len(input_ids))
print("First 16 token IDs:", input_ids[:16])
print("PS-001 present initially:", "PS-001" in rendered)
print("PS-002 present initially:", "PS-002" in rendered)
for marker in ["<|im_start|>", "<tools>", "<tool_call>", "<tool_response>"]:
    print(repr(marker), "->", tokenizer.encode(marker, add_special_tokens=False))
```

Output:

```text
Initial token count: 299
First 16 token IDs: [151644, 8948, 198, 10253, 1172, 25869, 7424, 13, 356, 632, 3255, 28360, 323, 1584, 26826, 382]
PS-001 present initially: False
PS-002 present initially: False
'<|im_start|>' -> [151644]
'<tools>' -> [27, 15918, 29]
'<tool_call>' -> [151657]
'<tool_response>' -> [151665]
```

The marker examples prevent another shortcut: a visually distinctive delimiter is not necessarily one special token. Inspect the tokenizer rather than assuming every XML-like tag has a dedicated vocabulary entry.

At the model level, the action continuation has the same autoregressive form as other generated text:

\[
P_\theta(y_1,\ldots,y_m\mid C)
=\prod_jP_\theta(y_j\mid C,y_{<j}),
\]

where C includes the rendered task and tool declaration. Parameters learned during pretraining/post-training can support selecting a tool and generating its argument syntax. The declaration supplies the available interface in context; it does not update weights or install a callable inside the transformer. Our rendering/token checks do not establish that this model will select the clinically appropriate table or consistently emit valid arguments.

Each schema token is embedded and participates in causal attention like other visible context. Hidden states at generated positions can therefore depend on the tool name, description, allowed fields, and prior observations; the output head turns the final hidden state into next-token logits. There is no separate automatic “register this Python function” operation inside the network. A familiar name may benefit from learned associations, while a new name needs enough contextual description for the model to use it. Being present in attention context does not prove correct understanding.

For training, a supervised tool-call example can teach the desired name and argument tokens through next-token loss. In agentic RL, the sampled call tokens are policy actions whose likelihoods enter the chosen policy objective; returned clinical records are environment observations. They condition later actions, but should not be mistaken for actions sampled by the policy. The exact action-token mask is developed in lesson 5.

Schema-constrained decoding, if explicitly configured and supported, can constrain syntax. The runtime still owns execution and access validation. This demonstration enables no such decoding and performs no sampling.

General template mechanics are documented by [Hugging Face chat templates](https://huggingface.co/docs/transformers/en/chat_templating). The exact markers and token IDs above come from our pinned cached tokenizer, not a claim about every model.

## 7. Parse an explicitly hand-authored assistant continuation

The usual package import also initializes AgentLoop and currently fails in our remote environment because Pillow (`PIL`) is absent. We load the unchanged, pinned parser source directly with `importlib` to isolate this component; no dependencies are mocked and this does not verify the full AgentLoop.

To isolate the parser, we provide a valid candidate continuation ourselves. It is not an observed Qwen response. We select the pinned `hermes` parser because its JSON-within-tool-call-tags grammar matches this example; this does not claim that a complete rollout configuration has been deployed.

```python
import importlib.util
import sys
parser_path = ROOT / ".upstream/verl/verl/experimental/agent_loop/tool_parser.py"
spec = importlib.util.spec_from_file_location("nanoastra_lesson_tool_parser", parser_path)
parser_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parser_module
spec.loader.exec_module(parser_module)
ToolParser = parser_module.ToolParser
candidate_text = '<tool_call>\n{"name":"get_clinical_record","arguments":{"field":"performance_status"}}\n</tool_call>'
candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
parser = ToolParser.get_tool_parser("hermes", tokenizer)
assistant_text, calls = asyncio.run(parser.extract_tool_calls(candidate_ids, [tool.tool_schema]))
print("Hand-authored continuation:")
print(candidate_text)
print("Parsed call count:", len(calls))
print("Parsed call:", calls[0].model_dump())
args = json.loads(calls[0].arguments)
print("Decoded argument type:", type(args).__name__)
```

Output:

```text
Hand-authored continuation:
<tool_call>
{"name":"get_clinical_record","arguments":{"field":"performance_status"}}
</tool_call>
Parsed call count: 1
Parsed call: {'arguments': '{"field": "performance_status"}', 'name': 'get_clinical_record', 'tool_call_id': None}
Decoded argument type: dict
```

The parser recognizes the delimiters, extracts the function name, and normalizes the arguments into a JSON string. The runtime then parses that string into a dictionary for execution. Successful parsing is not proof that the function exists or the arguments are allowed. Those checks occur after parsing.

This parser does not supply a call ID for this grammar: the output shows `None`. Hosted Chat Completions examples use linked call IDs, but that does not mean every local model grammar emits one. Never manufacture a claim that the model generated an identifier when the runtime added it.

## 8. Execute through the actual BaseTool lifecycle

```python
async def execute_selected_call():
    registry = {tool.name: tool}
    selected = registry[calls[0].name]
    key, initial = await selected.create(create_kwargs={"record": records})
    try:
        response, reward, metrics = await selected.execute(key, args)
        print("Creation revealed data:", not initial.is_empty())
        print("Response container:", type(response).__name__)
        print("Container field types:", {k: type(v).__name__ for k, v in response.model_dump().items()})
        print("Tool reward:", reward)
        print("Metrics:", metrics)
    finally:
        await selected.release(key)
    print("Remaining instances:", len(selected._records))
    return response, reward, metrics

response, reward, metrics = asyncio.run(execute_selected_call())
```

Output:

```text
Creation revealed data: False
Response container: ToolResponse
Container field types: {'text': 'str', 'image': 'NoneType', 'video': 'NoneType'}
Tool reward: 0.0
Metrics: {'record_queries': 1}
Remaining instances: 0
```

In this pinned verl version, `ToolResponse` has optional `text`, `image`, and `video` fields. The latter two are list-valued when used. Our text-only response leaves them unset. The return tuple also carries a numeric tool reward and a metrics dictionary. Neither is automatically embedded into the model-visible clinical text.

The adapter validates that parameters contain exactly `field`, checks the allowlist, and reads from its copied record store. `release` executes in a `finally` block so this demonstration leaves no instance behind. The actual loop's lifecycle and error handling are separate from the model's generated request.

## 9. Inspect the complete response text without projecting away fields

This block prints the exact string returned in `response.text`. No clinical fields are removed, and this is not a reconstructed `{scores: ...}` summary.

```python
print(response.text)
```

Output:

```text
{"status": "ok", "field": "performance_status", "value": "{\"patient_id\": \"SYN-ONC-001\", \"as_of\": \"2025-03-01T09:00:00Z\", \"table\": \"performance_status\", \"rows\": [{\"record_id\": \"PS-001\", \"version\": \"1\", \"patient_id\": \"SYN-ONC-001\", \"encounter_id\": \"ENC-001\", \"event_time\": \"2025-02-12T10:00:00Z\", \"available_time\": \"2025-02-12T10:00:00Z\", \"record_status\": \"active\", \"source_system\": \"SYNTHETIC_EHR\", \"scale\": \"ECOG\", \"score\": \"1\", \"assessor\": \"oncologist\", \"context\": \"ambulatory\"}, {\"record_id\": \"PS-002\", \"version\": \"1\", \"patient_id\": \"SYN-ONC-001\", \"encounter_id\": \"ENC-001\", \"event_time\": \"2025-02-28T14:00:00Z\", \"available_time\": \"2025-02-28T14:00:00Z\", \"record_status\": \"active\", \"source_system\": \"SYNTHETIC_EHR\", \"scale\": \"ECOG\", \"score\": \"2\", \"assessor\": \"oncologist\", \"context\": \"Ambulatory and capable of self-care; unable to work; up more than half of waking hours\"}]}"}
```

There are nested serialization layers:

| Layer | Python type | Contents |
| --- | --- | --- |
| `response` | verl `ToolResponse` | text/image/video container |
| `response.text` | `str` | JSON-encoded domain lookup result |
| `json.loads(response.text)` | `dict` | status, field, value |
| That dictionary's `value` | `str` | JSON-encoded table payload |
| `json.loads(value)` | `dict` | patient_id, as_of, table, rows |
| Each row | `dict` | Provenance and performance-status columns |

The extra JSON string inside `value` comes from reusing a generic record adapter whose values are text. It is not a universal API requirement or necessarily the ideal long-term output design. The current tool YAML declares no formal output JSON Schema, and the response container validates text type rather than the clinical meaning of each row.

## 10. Decode the complete clinical payload for readability

```python
domain_result = json.loads(response.text)
payload = json.loads(domain_result["value"])
print(json.dumps(payload, indent=2))
```

Output:

```text
{
  "patient_id": "SYN-ONC-001",
  "as_of": "2025-03-01T09:00:00Z",
  "table": "performance_status",
  "rows": [
    {
      "record_id": "PS-001",
      "version": "1",
      "patient_id": "SYN-ONC-001",
      "encounter_id": "ENC-001",
      "event_time": "2025-02-12T10:00:00Z",
      "available_time": "2025-02-12T10:00:00Z",
      "record_status": "active",
      "source_system": "SYNTHETIC_EHR",
      "scale": "ECOG",
      "score": "1",
      "assessor": "oncologist",
      "context": "ambulatory"
    },
    {
      "record_id": "PS-002",
      "version": "1",
      "patient_id": "SYN-ONC-001",
      "encounter_id": "ENC-001",
      "event_time": "2025-02-28T14:00:00Z",
      "available_time": "2025-02-28T14:00:00Z",
      "record_status": "active",
      "source_system": "SYNTHETIC_EHR",
      "scale": "ECOG",
      "score": "2",
      "assessor": "oncologist",
      "context": "Ambulatory and capable of self-care; unable to work; up more than half of waking hours"
    }
  ]
}
```

This is still the complete clinical table payload, merely decoded and pretty-printed. It contains both PS-001 and PS-002, the patient and encounter, event and availability timestamps, record versions/status, source system, ECOG scale, scores, assessor, and assessment context.

The row IDs are values of `record_id`, not necessarily top-level dictionary keys. Thus the representation is a **list of records**, not a map of `PS-001 -> score`, and not a score-only vector. Keeping the dates and source IDs makes longitudinal comparison and citation possible. The scores are strings because the CSV reader preserves text; a later typed conversion would be a separate explicit operation.

## 11. Demonstrate precisely what the earlier print statement did

```python
original_text = response.text
scores_only = [row["score"] for row in payload["rows"]]
print("Diagnostic projection only:", scores_only)
print("Original response unchanged:", response.text == original_text)
print("Row IDs still in full response:", all(rid in response.text for rid in ["PS-001", "PS-002"]))
print("Was a scores-only key returned:", "scores" in payload)
```

Output:

```text
Diagnostic projection only: ['1', '2']
Original response unchanged: True
Row IDs still in full response: True
Was a scores-only key returned: False
```

The list comprehension constructs a new list for display. It does not mutate `response.text`, `payload`, or the message content. If we intentionally sent only `json.dumps(scores_only)` to the model, then the IDs and context would indeed be absent. That would be a different tool-output design with different information loss; our current adapter does not do it.

Every future example should state whether it shows a full object, a selected-field projection, a template excerpt, or a hand-authored illustration. A shorter printout is useful only if its scope is explicit.

## 12. Construct the next messages from the parsed call and full response

For this local Qwen/Transformers example, we follow the pinned parser's output and keep arguments as a dictionary in the normalized assistant message. Since the parser produced no call ID, this single-call message pair has none. It is not presented as a Chat Completions HTTP request.

```python
assistant_message = {
    "role": "assistant", "content": assistant_text,
    "tool_calls": [{"type": "function", "function": {"name": calls[0].name, "arguments": args}}],
}
tool_message = {"role": "tool", "content": response.text}
updated_messages = messages + [assistant_message, tool_message]
print("Roles:", [message["role"] for message in updated_messages])
print("Tool content is exact response text:", tool_message["content"] == response.text)
print("Tool content includes both IDs:", "PS-001" in tool_message["content"] and "PS-002" in tool_message["content"])
print("Assistant call:", json.dumps(assistant_message["tool_calls"][0]))
```

Output:

```text
Roles: ['system', 'user', 'assistant', 'tool']
Tool content is exact response text: True
Tool content includes both IDs: True
Assistant call: {"type": "function", "function": {"name": "get_clinical_record", "arguments": {"field": "performance_status"}}}
```

The model-facing observation is the complete `response.text`, not the printed `scores_only` list. The call still appears in history so the result has conversational context.

Transformers tools and hosted API envelopes are related conventions, not interchangeable data structures in every implementation. [Hugging Face tool-use guide](https://huggingface.co/docs/transformers/en/chat_extras).

## 13. See the exact template block carrying the evidence

We render the updated conversation and print the complete newly relevant tool-response block, including its role marker and assistant continuation prefix. The earlier system/user/call portions are omitted from this **explicitly labeled template excerpt**, not from the actual `next_rendered` variable.

```python
next_rendered = tokenizer.apply_chat_template(updated_messages, tools=[schema], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
start = next_rendered.index("<|im_start|>user\n<tool_response>")
print("EXCERPT: complete tool-result block through assistant continuation prefix")
print(next_rendered[start:])
next_ids = tokenizer.encode(next_rendered, add_special_tokens=False)
print("Full updated token count:", len(next_ids))
print("Both record IDs survive rendering:", "PS-001" in next_rendered and "PS-002" in next_rendered)
```

Output:

```text
EXCERPT: complete tool-result block through assistant continuation prefix
<|im_start|>user
<tool_response>
{"status": "ok", "field": "performance_status", "value": "{\"patient_id\": \"SYN-ONC-001\", \"as_of\": \"2025-03-01T09:00:00Z\", \"table\": \"performance_status\", \"rows\": [{\"record_id\": \"PS-001\", \"version\": \"1\", \"patient_id\": \"SYN-ONC-001\", \"encounter_id\": \"ENC-001\", \"event_time\": \"2025-02-12T10:00:00Z\", \"available_time\": \"2025-02-12T10:00:00Z\", \"record_status\": \"active\", \"source_system\": \"SYNTHETIC_EHR\", \"scale\": \"ECOG\", \"score\": \"1\", \"assessor\": \"oncologist\", \"context\": \"ambulatory\"}, {\"record_id\": \"PS-002\", \"version\": \"1\", \"patient_id\": \"SYN-ONC-001\", \"encounter_id\": \"ENC-001\", \"event_time\": \"2025-02-28T14:00:00Z\", \"available_time\": \"2025-02-28T14:00:00Z\", \"record_status\": \"active\", \"source_system\": \"SYNTHETIC_EHR\", \"scale\": \"ECOG\", \"score\": \"2\", \"assessor\": \"oncologist\", \"context\": \"Ambulatory and capable of self-care; unable to work; up more than half of waking hours\"}]}"}
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>


Full updated token count: 715
Both record IDs survive rendering: True
```

The actual Qwen template wraps a `role=tool` message as a user-delimited block containing `<tool_response>`. This does not turn a tool observation into a human-authored request; it is this model's serialization convention. The template does not simply emit the literal Python role names verbatim for every role.

If a serving stack truncates the result or a context manager summarizes it, the effective evidence can differ. Checking this complete rendered example proves that our records survive this rendering; it does not prove every production prompt will preserve them or that a model will correctly use every field.

In the pinned verl AgentLoop, continuous-token helpers build the initial tokens and merge tool context into runtime IDs while aligning response masks/log-probabilities. Our full re-render here is an inspection exercise, not a substitute for that runtime merge or a proof that decode/re-encode preserves actual sampled IDs. Actual RL must retain and verify the real rollout token stream.

## 14. Check the failure contracts as well as the happy path

```python
async def invalid_calls():
    for parameters in [{"field": "mortality"}, {"field": "labs", "patient_id": "another-patient"}]:
        key, _ = await tool.create(create_kwargs={"record": records})
        try:
            error_response, _, _ = await tool.execute(key, parameters)
            print(json.dumps(parameters), "->", error_response.text)
        finally:
            await tool.release(key)
    key, _ = await tool.create(create_kwargs={"record": records})
    try:
        empty_response, _, _ = await tool.execute(key, {"field": "medication_administrations"})
        empty_result = json.loads(empty_response.text)
        print("Empty table:", empty_result["status"], json.loads(empty_result["value"])["rows"])
    finally:
        await tool.release(key)

asyncio.run(invalid_calls())
```

Output:

```text
{"field": "mortality"} -> {"status": "invalid_field"}
{"field": "labs", "patient_id": "another-patient"} -> {"status": "invalid_arguments"}
Empty table: ok []
```

`invalid_field` means the action requests a disallowed field. `invalid_arguments` means the supplied argument structure violates the adapter contract. A successfully read empty table yields `ok` with empty rows; that is not a negative clinical test or proof that no treatment occurred elsewhere. An infrastructure exception is yet another failure class.

The current domain contract also supports `unavailable` for an allowed field absent from the record dictionary. Oncology's projection normally creates every allowed table key, so missing observations within a table usually appear as empty rows or explicit pending records rather than that key-level status.

## 15. Which parts are standardized, and which are ours?

| Boundary | Relevant convention | Important limit |
| --- | --- | --- |
| Argument description | JSON Schema-style types/properties/required/enum | Supported subsets differ; description alone is not enforcement |
| OpenAI Chat Completions | assistant tool_calls, tool messages linked by tool_call_id | Different from Responses items |
| OpenAI Responses | function_call / function_call_output linked by call_id | Do not blindly reuse Chat message objects |
| Anthropic Messages | tool_use / tool_result content blocks | Results use its user-message convention |
| MCP client/server | Tool discovery and invocation, inputSchema and optional outputSchema | Does not impose one universal LLM prompt format |
| Local model rendering | Tokenizer/model chat template | Exact delimiters and metadata rendering are model-specific |
| nanoAstra clinical payload | status/field/value plus table records | Application-defined; no formal output-schema validator yet |

Official references checked for the protocol comparison: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [Anthropic tool handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls), and the versioned [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).

A formal output contract could later validate the success/error variants and typed row payloads. That would make software behavior more explicit; it would not guarantee that a clinical assertion is true or that the model has interpreted it correctly. We have not changed the payload format in this chapter because doing so silently would invalidate the very trace we are explaining.

## 16. Trace each step to the pinned implementation

| Step | Source location |
| --- | --- |
| Domain table projection | [oncology.py](../../src/nanoastra/oncology.py) |
| Field lookup result | [clinical.py](../../src/nanoastra/clinical.py) |
| Registry configuration | [oncology_tools.yaml](../../configs/oncology_tools.yaml) |
| Tool implementation | [verl_tool.py](../../src/nanoastra/verl_tool.py) |
| Framework response container | [pinned schemas.py](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/tools/schemas.py) |
| Hand-authored continuation parser | [pinned tool_parser.py](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_parser.py) |
| Schema passing, dispatch, result messages | [pinned tool_agent_loop.py](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/tool_agent_loop.py) |
| Continuous-token integration | [pinned agent_loop.py](https://github.com/verl-project/verl/blob/00094bd9cd3fef9cf8903daf60ea4a2bcf832efc/verl/experimental/agent_loop/agent_loop.py) |

The most important distinctions to retain are: declaration versus execution, full payload versus display projection, message structure versus rendered tokens, and environment observations versus sampled policy actions. Those distinctions determine both evidence integrity and later policy-loss masking.

Return to [lesson 3, runtime state](003-oncology-tools-and-environment.md#2-state-and-observation-are-different-objects). Verification and exact boundaries are recorded in [experiment 007](../../experiments/007-tool-schema-token-trace.md).
