[NARRATIVE GUARDRAIL — DO NOT IGNORE]
Numbers in this task come from a deterministic upstream computation.
Rules you MUST follow:
- Do not invent or recompute any number.
- Every dollar amount and percentage in your output must appear verbatim
  in the input data below.
- If a number is not in the input, write "[unknown]" instead of guessing.
- Pay attention to scale: 33,350 is "$33,350" (thirty-three thousand),
  not "$33.5 million".

[INPUT NUMBERS — copy these verbatim]
{{precomputed_numbers}}

[EXAMPLE — CORRECT BEHAVIOR]
Input: {"fund_summary": [{"fund": "General Operating", "total_budget": 8504000.0, "total_actual": 8537350.0, "net_variance_pct": 0.4}]}
Good: "General Operating Fund had a budget of $8,504,000 against actual of $8,537,350, a 0.4% net variance."
Bad:  "General Operating Fund showed a 0.4% variance against an $85.4 million budget." (wrong scale, invented number)

[EXAMPLE — CORRECT FLAGGED ITEM]
Input: {"line_item": "Adjunct Pay", "variance_dollars": 127450.0, "variance_pct": 18.6}
Good: "Adjunct Pay was over budget by $127,450 (18.6%)."
Bad:  "Adjunct Pay was over by roughly $127 million (the largest variance)."

[USER PROMPT FOLLOWS]
