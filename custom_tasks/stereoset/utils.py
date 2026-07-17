"""Helper functions for the StereoSet lm-evaluation-harness custom task.

Loaded automatically because the YAML uses !function utils.<name>.
"""


def stereoset_intrasentence_prompt(doc: dict) -> str:
    """Return the context sentence with BLANK preserved.

    The harness will fill BLANK with each choice returned by
    stereoset_intrasentence_choices() and score by log-likelihood.
    """
    return doc["context"]


def stereoset_intrasentence_choices(doc: dict) -> list[str]:
    """Return the three candidate completions in the order:
    [stereotype, anti-stereotype, unrelated].

    Downstream analysis code identifies which is which via the log-samples output.
    """
    sentences = doc["sentences"]["sentence"]
    labels = doc["sentences"]["gold_label"]
    # Sort so index 0 = stereotype, 1 = anti-stereotype, 2 = unrelated.
    order = {"stereotype": 0, "anti-stereotype": 1, "unrelated": 2}
    ordered = sorted(zip(sentences, labels), key=lambda p: order[p[1]])
    return [s for s, _ in ordered]


def stereoset_intersentence_prompt(doc: dict) -> str:
    return doc["context"]


def stereoset_intersentence_choices(doc: dict) -> list[str]:
    sentences = doc["sentences"]["sentence"]
    labels = doc["sentences"]["gold_label"]
    order = {"stereotype": 0, "anti-stereotype": 1, "unrelated": 2}
    ordered = sorted(zip(sentences, labels), key=lambda p: order[p[1]])
    return [s for s, _ in ordered]
