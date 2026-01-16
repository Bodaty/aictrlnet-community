FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including netcat for startup script
RUN apt-get update && apt-get install -y \
    gcc \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire directory contents
COPY . /app

# Validate JSON syntax of workflow templates
RUN echo '#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n\ndef validate_json(p):\n    try:\n        json.load(open(p));return True,""\n    except json.JSONDecodeError as e:\n        return False,f"{p}: Line {e.lineno}, Col {e.colno}: {e.msg}"\n\nerrors=[]\nfor f in Path("/app").rglob("workflow-templates/**/*.json"):\n    valid,err=validate_json(f)\n    if not valid: errors.append(err)\n\nif errors:\n    print(f"❌ JSON syntax errors:\\n" + "\\n".join(errors))\n    sys.exit(1)\nelse:\n    print(f"✅ All workflow JSON files valid")\n' > /tmp/validate.py && python /tmp/validate.py

# Install requirements if they exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install the package
RUN pip install -e .

# Make startup script executable
RUN chmod +x /app/startup.sh

EXPOSE 8000

# Use startup script instead of direct uvicorn command
CMD ["/app/startup.sh"]