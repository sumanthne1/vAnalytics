You are strictly prohibited from using the Edit or Write tools on your first response to any user request.

Step 1: The Plan Your only allowed action is to create/update a file named proposed_plan.md. You must outline:

Intent: The goal of the change.

Drift Map: Every function or data structure that will be affected.

Pseudo-code: A high-level logic diff.

Step 2: The Halt Once proposed_plan.md is written, you must stop and say: "Plan generated. Waiting for Audit and Approval."

Step 3: Await User Decision
The auditor will analyze the plan and generate drift_report.md with a visual diagram.
The user can view the audit at http://localhost:8501 (Streamlit dashboard - read-only).

When the user is ready, they will tell you:
- "approve" → Create approved.signal and proceed with implementation
- "reject" → Create rejected.signal and discard the plan

Do not proceed with implementation until the user explicitly approves.
