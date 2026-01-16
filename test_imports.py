#!/usr/bin/env python3
"""Test if models can be imported."""

import sys
sys.path.insert(0, '/app')

try:
    from src.models.community import Task, WorkflowDefinition, WorkflowInstance
    print("✓ Models imported successfully")
    print(f"Task: {Task}")
    print(f"WorkflowDefinition: {WorkflowDefinition}")
except Exception as e:
    print(f"✗ Failed to import models: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.api.v1.endpoints import tasks
    print("✓ Tasks endpoint imported successfully")
    print(f"Tasks router: {tasks.router}")
except Exception as e:
    print(f"✗ Failed to import tasks endpoint: {e}")
    import traceback
    traceback.print_exc()