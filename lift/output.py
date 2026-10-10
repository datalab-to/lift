import json
import logging

logger = logging.getLogger(__name__)


def load_output(output: str):
    try:
        return json.loads(output)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse model output as JSON: %r", output)
        return None
