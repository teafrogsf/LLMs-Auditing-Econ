import re

# Standard GSM8k ground-truth pattern
ANS_RE = re.compile(r"####\s*(\-?[0-9][0-9\,]*\.?[0-9]*)")

# Additional robust patterns for common model outputs
BOXED_RE = re.compile(r"\\boxed\{\s*(\-?[0-9][0-9\,]*\.?[0-9]*)\s*\}")
FINAL_ANSWER_PATTERNS = [
    re.compile(r"final\s*answer\s*[:：]?\s*(\-?[0-9][0-9\,]*\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"answer\s*is\s*[:：]?\s*(\-?[0-9][0-9\,]*\.?[0-9]*)", re.IGNORECASE),
    re.compile(r"答案(?:是|為)?\s*[:：]?\s*(\-?[0-9][0-9\,]*\.?[0-9]*)"),
    re.compile(r"最终答案\s*[:：]?\s*(\-?[0-9][0-9\,]*\.?[0-9]*)"),
]

INVALID_ANS = "[invalid]"




class GSM8kEvaluator:

    def extract_answer(completion):
        """
        Extract a numeric final answer from a model completion in a robust way.

        Priority:
        1) GSM8k-style: '#### <number>'
        2) LaTeX boxed: '\boxed{<number>}'
        3) Common phrasings: 'Final answer: <number>', 'Answer is <number>' (multilingual)
        4) Fallback to the last number appearing in the text
        """
        if completion is None:
            return INVALID_ANS

        text = completion if isinstance(completion, str) else str(completion)

        # 1) GSM8k-style
        match = ANS_RE.search(text)
        if match:
            return match.group(1).replace(",", "").strip()

        # 2) LaTeX boxed
        match = BOXED_RE.search(text)
        if match:
            return match.group(1).replace(",", "").strip()

        # 3) Common phrasings
        for pattern in FINAL_ANSWER_PATTERNS:
            found = pattern.search(text)
            if found:
                return found.group(1).replace(",", "").strip()

        # 4) Fallback: last number in the text
        numbers = re.findall(r"\-?[0-9][0-9\,]*\.?[0-9]*", text)
        if numbers:
            return numbers[-1].replace(",", "").strip()

        return INVALID_ANS


    def is_correct(model_completion, gt_example):
        gt_answer = GSM8kEvaluator.extract_answer(gt_example["answer"])
        assert gt_answer != INVALID_ANS
        return GSM8kEvaluator.extract_answer(model_completion) == gt_answer