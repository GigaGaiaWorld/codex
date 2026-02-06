# Parser Component

The **LangDa Parser** provides a unified, safe, and stateful parsing layer for detecting, validating, and structuring *extended ProbLog predicates*.
It supports both simple and full modes of special predicates (such as `langda(...)`), integrates with the multi-layer system (`Layer0–2`), and prepares clean intermediate representations for `create_graph(...)` and runtime execution.

---

## Overview

LangDa extends ProbLog syntax with **agentic predicates**, which can contain natural-language instructions, LLM/API specifications, and typed input/output definitions.
This parser is the front-end for such hybrid symbolic–neural predicates.

Core responsibilities:

* Tokenize and segment ProbLog source code into structured units (`Segment`)
* Recognize agentic predicates (`SimpleRecognizer`, `ChainRecognizer`)
* Parse predicate heads and arguments into type-safe structures (`PredRaw`)
* Distinguish **fixed**, **lazy**, and **dynamic** parameters

---

## Supported Predicate Forms

LangDa predicates can appear either **with** or **without** a logical head.

### 1. With Head

```prolog
weather(Cond, Wind) @langda(
    Agent:"Describe weather in natural language",
    IO:[+Cond:"condition", +Wind:"wind speed", -Out:"output text"],
    Models:[WeatherModel, "gpt-4.1"]
).
```

Rules:

* The head must follow standard ProbLog syntax:

  * Head name → problog predicate name format
  * Head variables → problog varible format (string or varname)
* The connection between the head and the agent predicate uses a decorator symbol:
  `head(...) @agent(...)`
* The parser recognizes such chains using `ChainRecognizer`.

### 2. Headless Form

```prolog
langda(
    Agent:"Standalone reasoning task",
    IO:[+X:"input var", -Y:"output var"],
    Models:["gpt-4.1"]
).
```

No head is defined; the agent itself represents a standalone logical node.
Parsed metadata (`PredRaw`) will have `head=None`.

---

## Argument Modes

Each agent predicate supports two parsing modes:

### 1. Simple Mode

Simple mode provides a compact three-slot shorthand form.
Each slot is **optional** (any subset of the three is valid), but **no slot may be repeated** and **no extra key–value pairs are allowed**.

#### Structure

```prolog
langda(
    weather,                           % template name (optional)
    [Model1, "gpt-4" | Cond, Wind],    % model + variable list (optional)
    "Describe the weather"             % agent instruction (optional)
).
```

* **Slot 1: HeadName**

  * Must match `^[A-Za-z_][A-Za-z0-9_]*$`
  * Defines the *target predicate name* to be generated, not a runtime argument.
* **Slot 2: Model/Variable list**

  * Must be enclosed in `[ ... ]`
  * With `|`: left = model list, right = variable list
    Without `|`: all elements are treated as variables
  * Model elements can be identifiers or quoted strings.
* **Slot 3: Agent string**

  * Must be a quoted string literal (`"..."` or `'...'`)
  * Defines the agent instruction (e.g. an LLM prompt)

Example combinations:

```prolog
langda("Describe weather").
langda(weather, "Describe weather").
langda([Model | Cond], "Describe weather").
langda(weather, [Model | Cond], "Describe weather").
```

Invalid examples:

```prolog
langda(weather, weather, "repeat").      % duplicate slot
langda(weather, Key:Val, "mixed").       % key–value pair not allowed here
```

---

### 2. Full Mode

Full mode allows explicit key–value arguments with strict typing and naming rules.

```prolog
langda(
    Name:"weather",
    Agent:"Describe weather in natural language",
    IO:[+Cond:"condition", +Wind:"wind speed", -Out:"response"],
    Models:[WeatherModel, "gpt-4.1"],
    Temperature:float,
    API:"https://weather.example.com"
).
```

#### Special Reserved Keys

| Key        | Description            | Value Type               | Regex Constraint                           |
| ---------- | ---------------------- | ------------------------ | ------------------------------------------ |
| **Name**   | Predicate name         | string (Name)            | `^[A-Za-z_][A-Za-z0-9_]*$`                 |
| **Agent**  | Instruction text       | string literal           | must be quoted                             |
| **IO**     | Input/output signature | `[+X:"desc", -Y:"desc"]` | variable keys: `^[+-]?[A-Z][A-Za-z0-9_]*$` |
| **Models** | List of LLM/API tools  | `[Model, "gpt-4.1"]`     | items = Name or string                     |

#### Generic Parameters

All other key–value pairs must satisfy:

* **Key:**
  `^_?[A-Z][A-Za-z0-9_]*$|^_$`
* **Value:**

  * A quoted string literal
  * Or a variable-like symbol (same regex as above)
* **Uniqueness:**

  * Each key must appear only once
* **Standalone key** (no `:`) → treated as `key: None` (flag parameter)

---

## Parameter Semantics

LangDa predicates distinguish **three categories of parameters** across system layers.

| Type                   | Layer                  | Description                                                                                                                                 |
| ---------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fixed Parameters**   | Layer 1 (create_graph) | Must always be defined; govern predicate generation. Includes `Agent`, `IO`, optionally `Name` and `Models`.                                |
| **Lazy Parameters**    | Layer 1 only           | Required during graph creation for typing or composition, but not exposed at runtime. Typically prefixed with `_`. Example: `_Weather:Api`. |
| **Dynamic Parameters** | Layer 2 (runtime)      | Exposed to runtime as inputs; injected dynamically by external data or user interaction. Example: `Weather:Api`.                            |

### Automatic Inference Rules

1. **`_Param` → lazy parameter**
   Exists only during graph creation.
2. **`Param` → dynamic parameter**
   Exposed at runtime.
3. **Agent placeholders** (`[[Param]]` or `__Param(...=>...)__`)
   Automatically create **lazy** parameters unless explicitly redefined as dynamic.
4. **Built-in lazy parameters**:
   `_hidden_func`, `_hidden_API` are always lazy but may internally depend on dynamic inputs.

---

## Multi-Layer Architecture

LangDa operates in **three layers** of interpretation.

### Layer 0 – Source Code (Syntax Layer)

User writes the predicate using ProbLog-style syntax:

```prolog
langda(
    Agent:"I want to ask __Question__ about Docs",
    IO:[+Question:"user question", +Docs:"document content", -Ans:"answer"],
    Docs:DocsType,
    Api:WeatherApi
).
```

The parser converts this into a structured form (`PredRaw`) with all arguments, placeholders, and metadata.

---

### Layer 1 – Graph Creation & Blueprint Definition

`create_graph(...)` combines the parsed predicate with a `blueprint.yaml` schema that defines each argument's role and type.

Example blueprint:

```yaml
predicates:
  langda:
    kind: agentic
    head_allowed: true

    slots:
      Agent:
        role: fixed
        required: true
        type: text
      IO:
        role: fixed
        type: io_signature
      Models:
        role: fixed
        type: model_list
      Weather:
        role: dynamic
        type: Api
      _Weather:
        role: lazy
        type: Api

    infer:
      from_agent_placeholders:
        pattern: "__([A-Za-z0-9_]+)__"
        default_role: lazy
        override_if_declared: true

    runtime_interface:
      expose:
        - Weather
        - Question
      rename:
        Question: question
```

**create_graph** behavior:

1. Parse LangDa source into a structured dict (`out_dict`).
2. Validate fixed slots (`Agent`, `IO`, etc.).
3. Identify lazy/dynamic slots via blueprint rules.
4. Construct the logical graph node (proxy predicate).
5. Output:

   * Expanded ProbLog/graph representation
   * Runtime schema describing which parameters must be provided dynamically.

---

### Layer 2 – Runtime Execution

Runtime executes the graph with dynamic parameters injected from external data sources.

Example runtime schema:

```yaml
runtime_predicates:
  langda_weather:
    inputs:
      - name: question
        type: text
      - name: weather
        type: Api
    outputs:
      - name: ans
        type: text
```

Two predicate types exist at runtime:

* **Agentic Predicate** — backed by an internal LLM or API agent; flexible input formats.
* **Static Predicate** — purely logical or deterministic; requires exact input structure (may support `update_graph` for adaptation).

---

## Comment Binding Rules

Comments inside langda predicates will be removed...

---

## Summary of Parsing Pipeline

1. **Lexical segmentation** – detect code, comments, predicates (`Segment`)
2. **Recognition** – activate recognizers (`SimpleRecognizer`, `ChainRecognizer`)
3. **Parsing** – convert to structured form (`PredRaw`, `OutDictType`)
4. **Annotation** – bind comments, attach placeholder and hash ID
5. **Integration** – feed structured output into `create_graph(...)`

---

## Example Integration

```python
from yourpkg.parser import UnifiedParser
from yourpkg.recognizer import SimpleRecognizer, ChainRecognizer
from yourpkg.predicate import PredicateParser
from yourpkg.models import PlaceholderStyle
from yourpkg.state import CodeState

source = """
weather(Cond, Wind) @langda(
    Agent:"Describe weather in natural language",
    IO:[+Cond:"condition", +Wind:"wind speed"],
    Models:["gpt-4.1"]
).
"""

parser = UnifiedParser(source)
parser.add_recognizer(SimpleRecognizer(["langda"]))
parser.add_recognizer(ChainRecognizer(["langda"]))
segments = parser.process()

for seg in segments:
    if seg.state == CodeState.PRED:
        pp = PredicateParser(seg.meta, placeholder_style=PlaceholderStyle.CURLY)
        out_dict, placeholder = pp.parse()
        print("Placeholder:", placeholder)
        print("Parsed:", out_dict)
```

---

## Design Philosophy

* **Clarity over magic:** the parser only interprets syntax; semantics live in blueprints.
* **Stage separation:**
  Layer 0 = source → structure
  Layer 1 = graph + blueprint → semantics
  Layer 2 = runtime → execution
* **Predictable evolution:** introducing new predicates only requires a new recognizer and YAML schema—no parser rewrites.

---

This `README.md` can serve as the official documentation for your `parser/` submodule in the LangDa project. It accurately represents the implemented syntax and seamlessly connects to your `create_graph` and `runtime` pipeline.