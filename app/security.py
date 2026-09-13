import re
from typing import Optional
from langsmith import traceable


class InputSanitizer:

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"forget\s+(all\s+)?(previous|prior|everything)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"do\s+not\s+(follow|obey)\s+(the\s+)?(previous|above)",
        r"new\s+instructions?\s*:",
        r"updated\s+instructions?\s*:",
        r"from\s+now\s+on\s*,?\s*you",
        r"your\s+new\s+(task|role|job)\s+is",
        r"system\s*prompt",
        r"system\s*message",
        r"reveal\s+(your|the)\s+(system|instructions|prompt)",
        r"show\s+me\s+(your|the)\s+(system|instructions|prompt)",
        r"what\s+(are|is)\s+your\s+(instructions|system\s*prompt)",
        r"print\s+(your|the)\s+(system|instructions|prompt)",
        r"repeat\s+(your|the)\s+(system|instructions|prompt)",
        r"---\s*end\s*(of)?\s*prompt",
        r"\[?\s*end\s+of\s+(system\s+)?prompt\s*\]?",
        r"###\s*(end|stop)",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"you\s+are\s+now\s+(DAN|jailbroken)",
        r"roleplay\s+as",
        r"imagine\s+you\s+(are|have)\s+no\s+(restrictions|rules)",
        r"simulate\s+(a|an)\s+(AI|assistant)\s+(without|with\s+no)",
        r"bypass\s+(all\s+)?restrictions",
        r"without\s+(any\s+)?(restrictions|limitations|filters)",
        r"no\s+(rules|restrictions|limitations)\s+apply",
        r"unlock\s+(developer|admin|god)\s*mode",
        r"developer\s+mode\s+(on|enabled|activated)",
        r"</?\s*system\s*>",
        r"</?\s*(user|assistant)\s*>",
        r"\[/?INST\]",
        r"<\|.*?\|>",
        r"translate\s+the\s+(above|previous)\s+.*\s+ignore",
        r"respond\s+only\s+in\s+(base64|rot13|hex)",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check(self, text: str):

        for pattern in self.patterns:
            if pattern.search(text):
                return False, "Blocked: potential prompt injection detected"

        return True, None

    def clean(self, text: str) -> str:

        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)
        text = text.replace("{{", "{ {").replace("}}", "} }")

        return text.strip()


class PIIDetector:

    PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    }

    MASK_MAP = {
        "email": "[EMAIL_REDACTED]",
        "phone": "[PHONE_REDACTED]",
        "ssn": "[SSN_REDACTED]",
        "credit_card": "[CREDIT_CARD_REDACTED]",
    }

    def detect(self, text: str):
        found = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = matches

        return found

    def mask(self, text: str):
        masked = text
        for pii_type, pattern in self.PATTERNS.items():
            masked = pattern.sub(self.MASK_MAP[pii_type], masked)
        return masked


class OutputValidator:

    HARMFUL_PATTERNS = [
        re.compile(r"here('s| is) (how|the way) to (hack|steal|attack)", re.I),
        re.compile(r"password\s+is\s+", re.I),
        re.compile(r"api[_\s]?key\s*[:=]", re.I),
    ]

    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output):
        if isinstance(output, list):
            output = "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in output])
        elif not isinstance(output, str):
            output = str(output)

        warnings = []

        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")

        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output = "[Response blocked: potetial harmful content]"
                warnings.append("Harmful content blocked")
                break

        return output, warnings


class SecurityPipeline:
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    @traceable(name="security check input")
    def check_input(self, text: str):
        notes = []

        is_safe, reason = self.sanitizer.check(text)

        if not is_safe:
            return False, "", [reason]

        cleaned = self.sanitizer.clean(text)

        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f"Input PII masked {list(pii_found.keys())}")

        return True, cleaned, notes

    @traceable(name="security check output")
    def check_output(self, text: str):
        return self.output_validator.validate(text)
