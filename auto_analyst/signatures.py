import dspy


class PlanAnalysis(dspy.Signature):
    """Create a concise analysis plan for the dataset and question."""

    question = dspy.InputField()
    dataset_overview = dspy.InputField()
    analysis_plan = dspy.OutputField(desc="Numbered steps describing analysis.")


class SelectTool(dspy.Signature):
    """Select the next tool and JSON input payload."""

    analysis_plan = dspy.InputField()
    available_tools = dspy.InputField()
    prior_results = dspy.InputField()
    tool_name = dspy.OutputField(desc="Exact tool name to call.")
    tool_input = dspy.OutputField(desc="JSON payload for the tool.")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize a final answer from tool results."""

    question = dspy.InputField()
    analysis_results = dspy.InputField()
    answer = dspy.OutputField(desc="Final response with insights and caveats.")
