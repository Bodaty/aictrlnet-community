#!/usr/bin/env python3
"""Tool Handler Validation Script.

This script validates that all registered tools have corresponding handler methods
in their respective tool dispatchers. Run this at startup or in CI to catch
missing handlers before they cause runtime AttributeError.

Usage:
    python scripts/test_tool_handlers.py

From Docker:
    docker exec dev-community-1 python scripts/test_tool_handlers.py
    docker exec dev-business-1 python scripts/test_tool_handlers.py
"""

import sys
import os

# Add paths for cross-edition imports
sys.path.insert(0, '/workspace/aictrlnet-fastapi/src')
sys.path.insert(0, '/workspace/aictrlnet-fastapi-business/src')

# For local development
local_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(local_path, 'src'))


def validate_community_tools():
    """Validate Community edition tool handlers."""
    print("\n=== Validating Community Edition Tools ===")

    try:
        from services.tool_dispatcher import CORE_TOOLS, ToolDispatcher, Edition

        # Create a minimal mock session for dispatcher instantiation
        class MockSession:
            pass

        # Get the tool handler map from the class
        handler_map = ToolDispatcher.TOOL_HANDLER_MAP

        print(f"Registered tools: {len(CORE_TOOLS)}")
        print(f"Handler mappings: {len(handler_map)}")

        missing = []
        for tool_name in CORE_TOOLS.keys():
            handler_name = handler_map.get(tool_name)
            if not handler_name:
                missing.append({
                    "tool": tool_name,
                    "issue": "no_mapping",
                    "message": f"Tool '{tool_name}' has no handler mapping"
                })
            elif not hasattr(ToolDispatcher, handler_name):
                missing.append({
                    "tool": tool_name,
                    "handler": handler_name,
                    "issue": "no_method",
                    "message": f"Tool '{tool_name}' handler '{handler_name}' not found"
                })

        if missing:
            print(f"\n❌ COMMUNITY VALIDATION FAILED: {len(missing)} issues")
            for item in missing:
                print(f"   - {item['message']}")
            return False
        else:
            print(f"\n✓ All {len(CORE_TOOLS)} Community tool handlers validated")
            return True

    except ImportError as e:
        print(f"Could not import Community tools: {e}")
        return False


def validate_business_tools():
    """Validate Business edition tool handlers."""
    print("\n=== Validating Business Edition Tools ===")

    try:
        from aictrlnet_business.services.tool_dispatcher import (
            AICTRLNET_TOOLS,
            BUSINESS_TOOLS,
            ENTERPRISE_TOOLS,
            ToolDispatcher,
            Edition
        )

        # Get all handler maps
        community_map = ToolDispatcher.TOOL_HANDLER_MAP
        business_map = ToolDispatcher.BUSINESS_TOOL_HANDLER_MAP
        enterprise_map = ToolDispatcher.ENTERPRISE_TOOL_HANDLER_MAP

        all_maps = {**community_map, **business_map, **enterprise_map}

        print(f"Total registered tools: {len(AICTRLNET_TOOLS)}")
        print(f"  - Community: {len([t for t in AICTRLNET_TOOLS.values() if 'community' in t.editions])}")
        print(f"  - Business: {len(BUSINESS_TOOLS)}")
        print(f"  - Enterprise: {len(ENTERPRISE_TOOLS)}")
        print(f"Handler mappings: {len(all_maps)}")

        missing = []
        for tool_name in AICTRLNET_TOOLS.keys():
            handler_name = all_maps.get(tool_name)
            if not handler_name:
                missing.append({
                    "tool": tool_name,
                    "issue": "no_mapping",
                    "message": f"Tool '{tool_name}' has no handler mapping"
                })
            elif not hasattr(ToolDispatcher, handler_name):
                missing.append({
                    "tool": tool_name,
                    "handler": handler_name,
                    "issue": "no_method",
                    "message": f"Tool '{tool_name}' handler '{handler_name}' not found"
                })

        if missing:
            print(f"\n❌ BUSINESS VALIDATION FAILED: {len(missing)} issues")
            for item in missing:
                print(f"   - {item['message']}")
            return False
        else:
            print(f"\n✓ All {len(AICTRLNET_TOOLS)} Business tool handlers validated")
            return True

    except ImportError as e:
        print(f"Could not import Business tools: {e}")
        print("  (This is expected if running in Community-only environment)")
        return True  # Not a failure, just not available


def main():
    """Run validation for all editions."""
    print("=" * 60)
    print("TOOL HANDLER VALIDATION")
    print("=" * 60)

    results = {
        "community": validate_community_tools(),
        "business": validate_business_tools(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for edition, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {edition.title()}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All tool handler validations passed!")
        return 0
    else:
        print("\n❌ Some validations failed - see above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
