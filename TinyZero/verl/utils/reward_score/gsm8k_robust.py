# adapted/created for RL COURSE at MIMUW

# better response extractor as the qwen is too dumb to type the "###" before a numeric answer
# same extractor as in phase 1

import re


def extract_last_number(text: str) -> str:
    if not text:
        return None

    def normalize_number(num_str: str) -> str:
        num_str = num_str.replace(',', '').strip()
        if num_str.endswith('.'):
            num_str = num_str[:-1]
        try:
            num = float(num_str)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except ValueError:
            return None

    def extract_from_snippet(snippet: str) -> str:
        boxed_match = re.search(r'\\boxed{([^}]+)}', snippet)
        if boxed_match:
            inner_matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', boxed_match.group(1))
            if inner_matches:
                normalized = normalize_number(inner_matches[-1])
                if normalized is not None:
                    return normalized

        parentheses_matches = re.findall(r'\\\((.*?)\\\)', snippet)
        if parentheses_matches:
            inner_matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', parentheses_matches[-1])
            if inner_matches:
                normalized = normalize_number(inner_matches[-1])
                if normalized is not None:
                    return normalized

        conclusion_match = re.search(r'(?i)(?:therefore|thus|hence|so,|the answer is|total is)(.*)', snippet)
        if conclusion_match:
            numbers_in_conclusion = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?(?!\s*%)', conclusion_match.group(1))
            if numbers_in_conclusion:
                normalized = normalize_number(numbers_in_conclusion[0])
                if normalized is not None:
                    return normalized

        cleaned_snippet = snippet.replace(',', '')
        tokens = re.split(r'[\s=\(\):;?!]+', cleaned_snippet)
        for token in reversed(tokens):
            if token.endswith('.'):
                token = token[:-1]
            try:
                num = float(token)
                if num.is_integer():
                    return str(int(num))
                return str(num)
            except ValueError:
                continue
        return None

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return None
    result = extract_from_snippet(lines[-1])
    if result is not None:
        return result
    return extract_from_snippet(text)


def compute_score(solution_str, ground_truth, score=1.0, format_score=0.0):
    pred = extract_last_number(solution_str)
    gt = str(ground_truth).replace(',', '').strip()
    return score if (pred is not None and pred == gt) else format_score
