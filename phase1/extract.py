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
        # 1: \boxed{number}
        boxed_match = re.search(r'\\boxed{([^}]+)}', snippet)
        if boxed_match:
            inner_matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', boxed_match.group(1))
            if inner_matches:
                normalized = normalize_number(inner_matches[-1])
                if normalized is not None:
                    return normalized

        # 2: \(number\)
        parentheses_matches = re.findall(r'\\\((.*?)\\\)', snippet)
        if parentheses_matches:
            inner_matches = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', parentheses_matches[-1])
            if inner_matches:
                normalized = normalize_number(inner_matches[-1])
                if normalized is not None:
                    return normalized

        # 3: Conclusion Parsing
        conclusion_match = re.search(r'(?i)(?:therefore|thus|hence|so,|the answer is|total is)(.*)', snippet)
        if conclusion_match:
            conclusion_text = conclusion_match.group(1)
            numbers_in_conclusion = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?(?!\s*%)', conclusion_text)
            if numbers_in_conclusion:
                normalized = normalize_number(numbers_in_conclusion[0])
                if normalized is not None:
                    return normalized

        # 4: Fallback (Reverse split)
        cleaned_snippet = snippet.replace(',', '')
        tokens = re.split(r'[\s=\(\):;?!$]+', cleaned_snippet)
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

    # last line 
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return None

    last_line = lines[-1]

    result = extract_from_snippet(last_line)
    if result is not None:
        return result

    # fallback whole text
    return extract_from_snippet(text)
