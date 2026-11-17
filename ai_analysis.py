import os
import requests
import re
from dotenv import load_dotenv
import zlib

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing. Add in .env file.")


def _normalize_req_type(req_type: str) -> str:
    t = str(req_type).strip().lower()
    if t in ("functional", "fr"):
        return "fr"
    if t in ("non-functional", "non functional", "nfr"):
        return "nfr"
    return t


def analyze_requirement(requirement: str, req_type: str):
    norm_type = _normalize_req_type(req_type)

    # requirement type -> UML diagram map
    diagram_map = {
        "fr": "Use Case Diagram",
        "nfr": "Component Diagram"
    }

    diagram_type = diagram_map.get(norm_type, "Component Diagram")

    prompt = f"""
    Analyze the following software requirement and generate:
    1. A user story with the format "As [role], I want to [action] so that [goal]".
    2. A list of software modules/components to be developed.
    3. An architectural base diagram in PlantUML, using components, relations and users.

    Requirement type: {req_type}
    Requirement: "{requirement}"

    The diagram must be a valid PlantUML {diagram_type}.

    Strict PlantUML syntax rules:
    - NEVER use the keyword `class`.
    - For components, ALWAYS use the format: component "Component Name" as Alias <<Stereotype>>
    - Always place `as Alias` BEFORE `<<Stereotype>>`.
    - For actors, ALWAYS use: actor "Actor Name" as Alias
    - For use cases, ALWAYS use: usecase "Use Case Name" as Alias

    Answer in format:
    ---
    User story:
    ...
    Components:
    - ...
    - ...
    PlantUML:
    ```plantuml
    @startuml
    ...
    @enduml
    ```
    """

    print("🧠 Call to Groq API...")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    r = requests.post(url, headers=headers, json=data)

    if r.status_code != 200:
        raise RuntimeError(f"Groq API error: {r.status_code} - {r.text}")

    output_text = r.json()["choices"][0]["message"]["content"]
    output_text = clean_output_text(output_text)

    # extraction of PlantUML block
    plantuml_code = extract_plantuml_block(output_text)

    if plantuml_code:
        plantuml_code = sanitize_plantuml(plantuml_code, norm_type)
    else:
        plantuml_code = '@startuml\nnote "Diagramma non generato per questo requisito" as N1\n@enduml'

    return output_text, plantuml_code


def sanitize_plantuml(plantuml_code: str, req_type: str) -> str:
    """Sanitize PlantUML code, fixing common LLM syntax mistakes."""
    norm_type = _normalize_req_type(req_type)

    # ensures @startuml and @enduml
    if not plantuml_code.strip().startswith("@startuml"):
        plantuml_code = "@startuml\n" + plantuml_code
    if not plantuml_code.strip().endswith("@enduml"):
        plantuml_code = plantuml_code + "\n@enduml"

    # clean duplicate @startuml
    plantuml_code = re.sub(r'@startuml\s+@startuml', '@startuml', plantuml_code)

    # fix "class Name <<Stereo>> as Alias" -> 'component "Name" as Alias <<Stereo>>'
    plantuml_code = re.sub(
        r'^(\s*)class\s+("?[^"<\n]+"?)\s*(<<[^>]+>>)?\s*as\s+([A-Za-z0-9_]+)',
        lambda m: f'{m.group(1)}component "{m.group(2).strip(chr(34))}" as {m.group(4)}' + (f' {m.group(3)}' if m.group(3) else ''),
        plantuml_code,
        flags=re.MULTILINE
    )

    # convert any remaining leading "class " into "component "
    plantuml_code = re.sub(r'^(\s*)class\s+', r'\1component ', plantuml_code, flags=re.MULTILINE)

    # ensure "as Alias" comes before "<<Stereotype>>"
    plantuml_code = re.sub(r'(<<[^>]+>>)\s+as\s+([A-Za-z0-9_]+)', r'as \2 \1', plantuml_code)

    # if it's a functional requirement (use case diagram) -> return after base fixes
    if norm_type == "fr":
        return plantuml_code

    # if it's a non-functional requirement (component diagram) -> fix blocks
    if norm_type == "nfr":
        # adds quotation marks around components with spaces in the name
        plantuml_code = re.sub(
            r'component\s+([^\s"{]+(?:\s+[^\s"{]+)+)\s+as',
            lambda m: f'component "{m.group(1)}" as',
            plantuml_code
        )

        # fix invalid { ... } blocks after components -> convert into note
        plantuml_code = re.sub(
            r'component\s+([A-Za-z0-9_]+)(?:\s+as\s+"[^"]+")?\s*\{[^"]*"([^"]+)"[^}]*\}',
            lambda m: f'component {m.group(1)}\nnote right of {m.group(1)}: {m.group(2)}',
            plantuml_code,
            flags=re.DOTALL
        )

        # fix actor blocks with { note ... }
        plantuml_code = re.sub(
            r'actor\s+([A-Za-z0-9_]+)\s+as\s+"([^"]+)"\s*\{[^"]*"([^"]+)"[^}]*\}',
            lambda m: f'actor {m.group(1)} as "{m.group(2)}"\nnote right of {m.group(1)}: {m.group(3)}',
            plantuml_code,
            flags=re.DOTALL
        )

        # fix wrong "as ..." inside notes
        plantuml_code = re.sub(
            r'note right of ([^ ]+) as "[^"]+"',
            r'note right of \1',
            plantuml_code
        )

    return plantuml_code


def extract_plantuml_block(text: str) -> str | None:
    match = re.search(r"(@startuml[\s\S]*?@enduml)", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None


def clean_output_text(output_text: str) -> str:
    """Remove introduction and final description from AI Analysis output"""
    cleaned = re.split(r"(?i)user story:", output_text, maxsplit=1)
    if len(cleaned) > 1:
        output_text = "User story:" + cleaned[1]

    output_text = re.sub(r"This .*?diagram.*", "", output_text, flags=re.IGNORECASE | re.DOTALL)

    return output_text.strip()


def plantuml_encode(text):
    """Encode PlantUML text to the format expected by plantuml.com server."""
    data = text.encode('utf-8')
    compressed = zlib.compress(data)[2:-4]  # strip zlib header/footer
    encoded = encode64(compressed)
    return encoded


def encode64(data):
    """PlantUML specific base64 encoding."""
    res = ""
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    for i in range(0, len(data), 3):
        if i + 2 == len(data):
            res += append3bytes(data[i], data[i + 1], 0, alphabet)
        elif i + 1 == len(data):
            res += append3bytes(data[i], 0, 0, alphabet)
        else:
            res += append3bytes(data[i], data[i + 1], data[i + 2], alphabet)
    return res


def append3bytes(b1, b2, b3, alphabet):
    c1 = b1 >> 2
    c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
    c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
    c4 = b3 & 0x3F
    return alphabet[c1] + alphabet[c2] + alphabet[c3] + alphabet[c4]





