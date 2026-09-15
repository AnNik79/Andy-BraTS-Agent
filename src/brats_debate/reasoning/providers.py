"""LLM provider abstraction. Secrets come from the environment, never from files."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


REQUIRED_FIELDS = (
    "overall_assessment",
    "expert_assessments",
    "likely_reliable_experts_by_region",
    "uncertainty_summary",
    "boundary_assessment",
    "unusual_region_flags",
    "reasoning",
    "limitations",
)


class LLMConfigurationError(ValueError):
    """Raised when real-LLM mode is requested without a usable provider/key."""


class LLMProviderError(RuntimeError):
    """Raised when a configured LLM provider fails. Never silently rewritten as LLM text."""


class LLMProvider:
    name = "base"

    def complete(self, evidence):
        raise NotImplementedError


def _validated(payload):
    if not isinstance(payload, dict):
        raise LLMProviderError("LLM provider must return a JSON object")
    missing = [key for key in REQUIRED_FIELDS if key not in payload]
    if missing:
        raise LLMProviderError(f"LLM response missing required fields: {missing}")
    return {key: payload[key] for key in REQUIRED_FIELDS}


def parse_model_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines[1:] if not line.strip().startswith("```"))
    try:
        return _validated(json.loads(text))
    except json.JSONDecodeError as exc:
        raise LLMProviderError("LLM response was not valid JSON") from exc


class DeterministicProvider(LLMProvider):
    """Explicit test/fallback mode. This is not an LLM and must not be labeled as one."""

    name = "deterministic"

    def complete(self, evidence):
        weights = evidence.get("controller_weights") or {}
        local = {name: (stats or {}).get("disagreement_region") for name, stats in weights.items()}
        local = {name: value for name, value in local.items() if value is not None}
        favored = max(local, key=local.get) if local else None
        regions = evidence.get("disagreement_regions") or []
        reliable = {}
        for region in regions:
            region_weights = region.get("controller_weights") or {}
            usable = {name: value for name, value in region_weights.items() if value is not None}
            if usable:
                reliable[region["id"]] = max(usable, key=usable.get)
            elif favored:
                reliable[region["id"]] = favored
        assessments = {}
        for name, role in (evidence.get("expert_roles") or {}).items():
            entropy = ((evidence.get("uncertainty") or {}).get(name) or {}).get("overall")
            assessments[name] = (
                f"{name} ({role}): coverage={(evidence.get('expert_coverage') or {}).get(name)}, "
                f"mean entropy={entropy}"
            )
        pair = evidence.get("expert_pair_with_largest_disagreement") or "none"
        mean = evidence.get("mean_disagreement")
        return {
            "overall_assessment": (
                f"Deterministic fallback summary: mean disagreement={mean}. "
                f"Largest pairwise probability distance is {pair}."
            ),
            "expert_assessments": assessments,
            "likely_reliable_experts_by_region": reliable,
            "uncertainty_summary": evidence.get("uncertainty") or {},
            "boundary_assessment": (
                f"Boundary-interface disagreement={evidence.get('boundary_disagreement')}. "
                "This uses predicted class interfaces, not a separate anatomy atlas."
            ),
            "unusual_region_flags": [flag["label"] for flag in evidence.get("unusual_or_difficult_flags") or []],
            "reasoning": (
                "Structured statistics were compiled from the four segmentation experts and the "
                "disagreement map. No language model was called."
            ),
            "limitations": evidence.get("interpretation_limit") or (
                "Deterministic fallback; not medical advice and not an LLM interpretation."
            ),
        }


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, api_key, model, base_url="https://api.openai.com/v1", timeout=60):
        if not api_key:
            raise LLMConfigurationError("OpenAI-compatible provider requires an API key from the environment")
        if not model:
            raise LLMConfigurationError("OpenAI-compatible provider requires a model name")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, evidence):
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": (
                    "You are the LLM reasoning expert in a research-only BraTS segmentation debate. "
                    "You reason over structured evidence from four segmentation experts and their disagreement. "
                    "You must not invent voxel masks, medical diagnoses, or anatomy that is not in the evidence. "
                    "Return only a JSON object with keys: overall_assessment, expert_assessments, "
                    "likely_reliable_experts_by_region, uncertainty_summary, boundary_assessment, "
                    "unusual_region_flags, reasoning, limitations."
                )},
                {"role": "user", "content": json.dumps(evidence, sort_keys=True)},
            ],
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise LLMProviderError(f"LLM HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise LLMProviderError(f"LLM request failed: {exc.reason}") from None
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM response missing message content") from exc
        return parse_model_json(text)


def provider_from_config(cfg):
    reasoning = cfg.get("reasoning") or {}
    mode = reasoning.get("mode", "deterministic")
    if mode == "deterministic":
        return DeterministicProvider(), "deterministic"
    if mode != "llm":
        raise LLMConfigurationError("reasoning.mode must be 'llm' or 'deterministic'")
    key = os.environ.get(reasoning.get("api_key_env", "BRATS_LLM_API_KEY") or "", "").strip()
    model = (reasoning.get("model") or os.environ.get(reasoning.get("model_env", "BRATS_LLM_MODEL") or "") or "").strip()
    base = (os.environ.get(reasoning.get("base_url_env", "BRATS_LLM_BASE_URL") or "") or "https://api.openai.com/v1").strip()
    if not key:
        raise LLMConfigurationError(
            "Real LLM reasoning was requested (reasoning.mode=llm) but the API key environment "
            f"variable {reasoning.get('api_key_env', 'BRATS_LLM_API_KEY')} is unset. "
            "Set the key, or set reasoning.mode: deterministic for explicit test fallback."
        )
    if not model:
        raise LLMConfigurationError(
            "Real LLM reasoning was requested but no model is configured. Set reasoning.model "
            f"or {reasoning.get('model_env', 'BRATS_LLM_MODEL')}."
        )
    timeout = int(reasoning.get("timeout_seconds", 60))
    name = reasoning.get("provider", "openai_compatible")
    if name != "openai_compatible":
        raise LLMConfigurationError(f"Unknown reasoning.provider: {name}")
    return OpenAICompatibleProvider(key, model, base, timeout), "llm"
