# Interactive lesson 3: evidence access and the agent boundary

Date: 2026-09-29. This is a teaching summary, not a verbatim transcript.

## User request, translated

Start explaining lesson 3 interactively.

## Step 1: one query, one observation

The tutor starts with the existing oncology snapshot. The environment holds thirteen filtered clinical tables; calling `read_record` for `performance_status` returns only that table. The returned record IDs/scores are PS-001/1 and PS-002/2. These values were checked by running the repository code locally.

The distinction is between information held by the environment and information actually serialized into the model's messages. A Python variable containing all records is not automatically visible to the LLM. The current snippet scripts the action; a future policy would generate the tool request. No model inference or training occurs in this step.

The acquisition loop is: model selects a table, the environment returns evidence, and the model uses the expanded history for the next decision. Subsequent steps will introduce runtime state, budget, transitions, and reset after checking understanding of this boundary.

## Pending question

If the backend has all thirteen tables but the model has only requested performance status, has the model also seen the laboratory and note rows? Explain what determines visibility. No learner answer has been received yet.

## Learner response and step 2

Translated learner response: No; visibility depends on the model's historical tool calls.

Tutor clarification: tool-call history is a useful starting point, but effective visibility depends on the actual context sent to the model. A requested table is not necessarily successfully returned; a returned result may not be appended to model messages, or may later be truncated/summarized. History includes both actions and observations. Backend logs are not equivalent to model input.

A local teaching trace ran the existing performance-status lookup, showing roles before result insertion as `[user, assistant]` and afterward as `[user, assistant, tool]`. This is a conceptual message trace, not a model-specific chat template or actual LLM call. No inference ran.

Next understanding check: if a record query leaves the patient and database unchanged but adds evidence and consumes a query budget unit, has the environment's runtime state changed? The learner has not answered this question yet.

## Follow-up: tool definitions and protocol standards

Translated learner questions: How is a tool defined? Is there a common schema? Is there a standard message structure?

The tutor distinguishes model-visible schema, runtime implementation, message envelope, and final model-specific token rendering. Local YAML and BaseTool code were inspected, and official OpenAI, Anthropic, and versioned MCP documentation were checked. JSON Schema is shared terminology for inputs, but message/call-result formats vary. The previous assistant-content example is explicitly clarified as conceptual rather than a complete API tool call.

Examples and source links are saved in [the protocol supplement](../notes/tool-schemas-and-messages.md). Its JSON is hand-authored configuration/protocol illustration, not observed model output or an API test. The pending interactive question asks who actually executes the Python function after a model requests it.

## Continue: request, dispatch, and return

The learner asked to continue. The tutor explained that schema alone cannot execute code and demonstrated a hand-authored call dispatched to the real local lookup. The script printed the parsed field, executed tool name, two performance-status record scores, matched call ID, and user/assistant/tool roles for the next request. No API or model was invoked.

Teaching checkpoint: distinguish a model-selected request from environment-produced values. The next question asks whether the returned scores count as model-generated actions merely because they later enter the model context. No learner answer is recorded yet.

## Follow-up: response schema

The learner asked what the tool response schema looks like. The tutor inspected the pinned local verl ToolResponse source and nanoAstra adapter, then ran the domain lookup. The explanation distinguishes the framework's text/image/video container, the adapter's response/reward/metrics tuple, the application-defined status/field/value payload, and the model-message envelope. It explicitly notes the current nested JSON string in value and the absence of a declared output JSON Schema in the tool YAML. No model or API was run.

## Follow-up: complete records and a dedicated chapter

Translated learner clarification: Is the actual result just `scores: [1, 2]`, rather than records such as PS-001? The tutor clarified that scores were a display projection; the actual response retains full rows and provenance.

Translated learner request: These details are important and need a dedicated chapter explaining schemas, tool definitions, and how the model receives them.

Delivered [Lesson 3A](../lessons/003a-tool-schemas-to-model-tokens.md), with 13 executed examples covering the real YAML/schema, cached Qwen template and token IDs, an explicitly hand-authored continuation, the real pinned parser, BaseTool lifecycle, complete nested JSON result, message insertion, and error contracts. All 47 current curriculum snippets were executed. No model inference ran. The parser was loaded directly from its pinned source because the package's AgentLoop initializer requires the absent Pillow dependency; full AgentLoop execution remains unverified.

## Follow-up: where the record dictionary comes from

Translated learner question: Why is record a dictionary, and which step converts it?

Source-checked explanation: `csv.DictReader` creates row dictionaries; `snapshot` groups filtered row lists by table; `tool_view` builds a table-to-JSON-string dictionary. The caller passes that existing dictionary as `create_kwargs["record"]`; the adapter's `dict(source)` only copies it. This application-specific lookup contract differs from both a CSV row and the dictionary reconstructed by `json.loads` on a response. Added this distinction to Lesson 3A section 4; no executable examples changed.
