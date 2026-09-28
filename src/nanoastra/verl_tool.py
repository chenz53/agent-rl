"""Actual verl BaseTool adapter, smoke-tested against the pinned verl on vp.

Install a compatible verl environment before importing this module. There is no
mock fallback: missing dependencies should fail rather than simulate integration.
"""

import json
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from nanoastra.clinical import ALLOWED_FIELDS, read_record


class ClinicalRecordTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        fields = config.get("record_fields", list(ALLOWED_FIELDS))
        if not isinstance(fields, list) or not fields or not all(isinstance(x, str) for x in fields):
            raise ValueError("record_fields must be a nonempty list of field names.")
        if set(fields) & {"label", "ground_truth", "instructor_only", "reference_diagnosis"}:
            raise ValueError("Answer keys are not queryable record fields.")
        self._allowed_fields = tuple(fields)
        self._records = {}

    async def create(self, instance_id=None, **kwargs):
        # The pinned ToolAgentLoop calls create(create_kwargs=...) per tool call.
        source = kwargs["create_kwargs"]["record"]
        if not isinstance(source, dict) or set(source) - set(self._allowed_fields):
            raise ValueError("Tool context must contain only permitted record fields.")
        if not all(isinstance(value, str) for value in source.values()):
            raise ValueError("Record values must be strings.")
        key = instance_id or str(uuid4())
        if key in self._records:
            raise ValueError("Tool instance ID already exists.")
        self._records[key] = dict(source)
        # Returning the whole record here would reveal information prematurely.
        return key, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        if not isinstance(parameters, dict) or set(parameters) != {"field"}:
            result = {"status": "invalid_arguments"}
        else:
            result = read_record(self._records[instance_id], parameters["field"], self._allowed_fields)
        # Zero tool reward: terminal correctness is scored separately.
        # No query-cost penalty until its end-to-end aggregation is verified.
        return ToolResponse(text=json.dumps(result)), 0.0, {"record_queries": 1}

    async def release(self, instance_id, **kwargs):
        self._records.pop(instance_id, None)
