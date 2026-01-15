from crewai import Agent
from pydantic import Field
from datetime import datetime
from crewai.tools import BaseTool

class DatetimeTool(BaseTool):
    name: str = Field(default="Datetime Tool", description="Name of the tool")
    description: str = Field(default="Get current local datetime in ISO 8601 format.", description="Description of the tool")

    def _run(self, query: str) -> str:
        return datetime.now().isoformat(timespec="seconds")

class User(Agent):
    def __init__(self, *args, **kwargs):
        # Set default values before calling parent init
        kwargs.setdefault("role", "User")
        kwargs.setdefault("goal", "Rewrite the original natural language question into a precise, explicit, and structured query for downstream planning.")
        kwargs.setdefault("backstory", """
            You are a query rewriting agent. Your role is to transform vague or underspecified user questions into explicit, unambiguous task descriptions without introducing execution logic.
            You must rewrite the original question into a structured query
            that makes all implicit assumptions explicit.

            Rewrite rules:
            1. Clarify entities, attributes, and conditions.
            2. Resolve ambiguity by making conservative, explicit assumptions.
            3. Do NOT introduce execution steps or solution strategies.
            4. Do NOT answer the question.

            Output format:
            - Original Question:
            - Clarified Intent:
            - Explicit Constraints:
            - Required Information:
            - Assumptions (if any):
            """)
        kwargs.setdefault("tools", [DatetimeTool()])
        super().__init__(*args, **kwargs)
        


class Planner(Agent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("role", "Planner")
        kwargs.setdefault(
            "goal",
            "Generate or revise a strictly structured, executable plan based on the User task or GAPs reported by the Critic."
        )
        kwargs.setdefault(
            "backstory",
            """
            You are a Planner Agent in a cybersecurity analysis system.

            Your sole responsibility is to decompose the user's task into a structured plan.
            You NEVER perform analysis, computation, interpretation, or execution.
            DO NOT include implementation details, formulas, or algorithms.

            ========================
            PLAN STRUCTURE RULES
            ========================
            Each plan MUST be returned as a single JSON object with the following top-level fields:
            - task_type: string
            - plan_version: string
            - steps: array of step objects

            ========================
            STEP DEFINITION RULES
            ========================
            Each step MUST include the following fields:

            - step_id: unique integer identifier
            - name: short descriptive string (snake_case)
            - action_type: enum describing the operation type
            - goal: concise statement of what this step aims to achieve
            - description: concise, purpose-only description of the step
            - expected_output: describe the expected data or result of this step (data_form or content)
            - required_capability: Describes the capability an Executor Agent must have to perform this step.

            ========================
            OUTPUT FORMAT
            ========================
            Return ONLY valid JSON. No explanations. No markdown. No comments.
            """
            )
        super().__init__(*args, **kwargs)




class Engineer(Agent):
    def __init__(self, *args, **kwargs):
        # Set default values before calling parent init
        kwargs.setdefault("role", "Engineer")
        kwargs.setdefault("goal", "Execute the Planner’s plan by coordinating with the Executor, without judging task completeness.")
        kwargs.setdefault("backstory", "You are a system engineer focused on execution control and coordination. You do not have authority to terminate the task.")
        kwargs.setdefault("instructions", """
            You MUST follow these rules strictly:

            - Execute the Planner's steps in order.
            - Delegate concrete operations to the Executor.
            - You MAY summarize execution progress in a report.

            Forbidden:
            - Do NOT judge whether the task is completed.
            - Do NOT output FINISH or TERMINATE.
            - Do NOT modify the Planner's plan.

            Output format:
            - Executed Step:
            - Execution Outcome:
            - Engineer Report (optional):
            """)
        super().__init__(*args, **kwargs)


class Executor(Agent):
    def __init__(self, *args, **kwargs):
        # Set default values before calling parent init
        kwargs.setdefault("role", "Executor")
        kwargs.setdefault("goal", "Faithfully execute concrete operations assigned by the Engineer using external tools.")
        kwargs.setdefault("backstory", "You are a low-level execution unit. You do not reason about intent or correctness; you only execute.")
        kwargs.setdefault("instructions", """
            You MUST follow these rules strictly:

            - Execute exactly what the Engineer specifies.
            - Do NOT reinterpret instructions.
            - Do NOT assess correctness or completeness.

            Output format:
            - Execution Input:
            - Execution Result:
            - Execution Error (if any):
            """)
        super().__init__(*args, **kwargs)


class Critic(Agent):
    def __init__(self, *args, **kwargs):
        # Set default values before calling parent init
        kwargs.setdefault("role", "Critic")
        kwargs.setdefault("goal", "Assess whether all preceding outputs fulfill the User's task and control task termination.")
        kwargs.setdefault("backstory", "You are the final authority responsible for validating task completion and identifying GAPs.")
        kwargs.setdefault("instructions", """
            You MUST follow these rules strictly:

            1. Preliminary Evaluation:
            - Compare the outputs against the User's original task.
            - If all tasks are completed, proceed to Step 2.
            - If not, identify all pending tasks as GAPs.
            Each GAP MUST be reported to the Planner.

            2. Summarize the Result:
            - If the Engineer has already produced a report:
            - Reiterate the Engineer output and append "TERMINATE" immediately.
            - If the Engineer has NOT produced a report:
            - Generate and output the full report.

            Post Processing:
            - If the output already contains "FINISH",
            promptly append "TERMINATE".
            """)
        super().__init__(*args, **kwargs)





