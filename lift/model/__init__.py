from typing import List

from lift.model.hf import load_model, generate_hf
from lift.model.schema import BatchInputItem, BatchOutputItem
from lift.model.vllm import generate_vllm
from lift.output import load_output
from lift.settings import settings


class InferenceManager:
    def __init__(self, method: str = "vllm"):
        assert method in ("vllm", "hf"), "method must be 'vllm' or 'hf'"
        self.method = method

        if method == "hf":
            self.model = load_model()
        else:
            self.model = None

    def generate(
        self, batch: List[BatchInputItem], max_output_tokens=None, **kwargs
    ) -> List[BatchOutputItem]:
        vllm_api_base = kwargs.pop("vllm_api_base", settings.VLLM_API_BASE)

        if self.method == "vllm":
            results = generate_vllm(
                batch,
                max_output_tokens=max_output_tokens,
                vllm_api_base=vllm_api_base,
                **kwargs,
            )
        else:
            results = generate_hf(
                batch,
                self.model,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )

        output = []
        for result, input_item in zip(results, batch):
            extraction = load_output(result.raw)
            output.append(
                BatchOutputItem(
                    raw=result.raw,
                    token_count=result.token_count,
                    error=result.error,
                    extraction=extraction
                )
            )
        return output
