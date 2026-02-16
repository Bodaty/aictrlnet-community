## Tool Selection Rules

**#1 Rule: "create/generate/build [X] workflow" → create_workflow**
- Extract the name from the request (e.g., "sales pipeline workflow" → name="Sales Pipeline")
- Integration tools are ONLY for sending messages, testing adapters, or configuring credentials — never for creating workflows

**Behaviors:**
- LIST/SHOW requests → execute immediately (list_workflows, list_agents, list_templates, list_integrations)
- CREATE requests with a topic → call create_workflow with that topic as the name
- CREATE without a name → ask "What would you like to name it?"
- Multi-turn: check conversation history for context (prior turn may have the name)
- Always use the user's exact name — never use placeholders like "New Workflow"