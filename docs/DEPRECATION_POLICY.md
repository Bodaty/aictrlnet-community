# AICtrlNet Deprecation Policy

This document outlines the deprecation policy for AICtrlNet Community Edition. It covers how features, APIs, and configurations are deprecated, the timeline for removals, and the support provided during transitions.

## Table of Contents

- [Versioning Scheme](#versioning-scheme)
- [Deprecation Timeline and Process](#deprecation-timeline-and-process)
- [API Deprecation Policy](#api-deprecation-policy)
- [Breaking Changes Policy](#breaking-changes-policy)
- [Support Timeline](#support-timeline)
- [Migration Assistance](#migration-assistance)

---

## Versioning Scheme

AICtrlNet follows [Semantic Versioning 2.0.0](https://semver.org/) (SemVer):

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Incremented for incompatible API changes or breaking changes
- **MINOR**: Incremented for new functionality added in a backward-compatible manner
- **PATCH**: Incremented for backward-compatible bug fixes

### Version Examples

| Version | Description |
|---------|-------------|
| `1.0.0` | Initial stable release |
| `1.1.0` | New features added, backward compatible |
| `1.1.1` | Bug fixes only |
| `2.0.0` | Breaking changes introduced |

### Pre-release Versions

Pre-release versions use suffixes:
- `-alpha.N`: Early development, unstable
- `-beta.N`: Feature complete, testing phase
- `-rc.N`: Release candidate, final testing

Example: `2.0.0-beta.1`

---

## Deprecation Timeline and Process

### How Features Are Marked as Deprecated

1. **Documentation Update**: The feature is marked as deprecated in all relevant documentation with a clear notice and recommended alternative.

2. **Code Warnings**: Deprecation warnings are emitted in logs and responses:
   ```python
   # Log warning
   logger.warning("Feature X is deprecated and will be removed in v2.0.0. Use Feature Y instead.")
   ```

3. **API Response Headers**: Deprecated API endpoints include warning headers:
   ```
   Deprecation: true
   Sunset: Sat, 01 Jan 2026 00:00:00 GMT
   Link: <https://docs.aictrlnet.com/migration/feature-x>; rel="deprecation"
   ```

4. **Changelog Entry**: Every deprecation is documented in the CHANGELOG with:
   - What is being deprecated
   - Why it is being deprecated
   - What to use instead
   - When it will be removed

### Minimum Notice Period

| Change Type | Minimum Notice | Notes |
|-------------|----------------|-------|
| Feature deprecation | 2 minor versions or 6 months (whichever is longer) | Ample time for migration |
| API endpoint deprecation | 2 minor versions or 6 months | Both old and new endpoints available |
| Configuration option deprecation | 1 minor version or 3 months | Warning logged when used |
| Security-related deprecation | Immediate in critical cases | Security takes precedence |

### Communication Channels

Deprecation notices are communicated through:

1. **Release Notes**: Every release includes a "Deprecated" section
2. **CHANGELOG.md**: Comprehensive deprecation history
3. **GitHub Issues**: Deprecation tracking issues with `deprecation` label
4. **Documentation**: Dedicated deprecation notices in docs
5. **In-App Warnings**: Runtime warnings in logs and API responses
6. **GitHub Discussions**: Community announcements for major deprecations

---

## API Deprecation Policy

### API Versioning

AICtrlNet uses URL-based API versioning:

```
/api/v1/workflows
/api/v2/workflows
```

### Version Lifecycle

| Phase | Duration | Description |
|-------|----------|-------------|
| **Current** | Ongoing | Latest stable version, receives all updates |
| **Supported** | 12 months after next major | Previous major version, receives bug fixes and security patches |
| **Deprecated** | 6 months | Marked for removal, no new features |
| **Removed** | - | No longer available |

### API Deprecation Process

1. **New Version Release**: When a new API version is released, the previous version enters "Supported" status.

2. **Deprecation Announcement**: 6 months before removal:
   - Documentation updated with deprecation notice
   - `Deprecation` header added to responses
   - Sunset date announced

3. **Warning Period**: During deprecation:
   - API continues to function normally
   - Responses include deprecation headers
   - Usage metrics tracked for impact assessment

4. **Removal**: After the sunset date:
   - Endpoint returns `410 Gone`
   - Response includes migration guidance

### Backward Compatibility Guarantees

Within a major version, we guarantee:

- Existing endpoints remain functional
- Response field additions only (no removals)
- Request parameters remain optional unless security-critical
- Error codes and formats remain consistent
- Authentication methods remain supported

### Endpoint Deprecation Example

```python
# Deprecated endpoint response
HTTP/1.1 200 OK
Deprecation: true
Sunset: Sat, 01 Jul 2026 00:00:00 GMT
X-Deprecated-Message: This endpoint is deprecated. Use /api/v2/workflows instead.
Link: </api/v2/workflows>; rel="successor-version"

{
  "data": { ... },
  "_deprecation": {
    "message": "This endpoint will be removed on 2026-07-01",
    "migration_guide": "https://docs.aictrlnet.com/migration/workflows-v2"
  }
}
```

---

## Breaking Changes Policy

### What Constitutes a Breaking Change

The following are considered breaking changes and require a major version bump:

**API Changes:**
- Removing an endpoint
- Removing a required request parameter
- Removing a response field
- Changing the type of an existing field
- Changing authentication requirements
- Changing error response formats

**Behavioral Changes:**
- Changing default values that affect existing behavior
- Modifying business logic that changes outcomes
- Changing database schema in incompatible ways
- Removing or renaming environment variables

**Dependency Changes:**
- Dropping support for a Python version
- Requiring a new system dependency
- Upgrading dependencies with breaking changes

### What Is NOT a Breaking Change

- Adding new optional parameters
- Adding new response fields
- Adding new endpoints
- Adding new optional features
- Bug fixes (even if users depended on buggy behavior)
- Performance improvements
- Documentation updates

### How Breaking Changes Are Handled

1. **Proposal**: Breaking changes are proposed via GitHub Issues with the `breaking-change` label

2. **Community Input**: 30-day comment period for community feedback

3. **Migration Path**: A clear migration guide must be prepared before implementation

4. **Implementation**: Breaking changes are batched into major releases when possible

5. **Transition Period**: Old behavior available during deprecation period

### Major Version Bump Requirements

A major version bump (`X.0.0`) is required when:

- Any breaking change is introduced
- Minimum Python version is increased
- Fundamental architecture changes occur
- Security model changes require user action

---

## Support Timeline

### Version Support Policy

| Version Type | Active Support | Security Support | Total Support |
|--------------|----------------|------------------|---------------|
| Latest Major | Full features, bug fixes | Security patches | Until next major + 6 months |
| Previous Major | Critical bug fixes only | Security patches | 12 months after next major |
| Older Versions | None | None | Unsupported |

### Security Patch Policy

- **Critical vulnerabilities (CVSS 9.0+)**: Patched within 48 hours for all supported versions
- **High vulnerabilities (CVSS 7.0-8.9)**: Patched within 7 days for all supported versions
- **Medium vulnerabilities (CVSS 4.0-6.9)**: Patched in next regular release
- **Low vulnerabilities (CVSS < 4.0)**: Addressed in future releases as prioritized

### End of Life (EOL) Process

1. **6 months before EOL**: Announcement in release notes and documentation
2. **3 months before EOL**: Warning banner in documentation
3. **1 month before EOL**: Final reminder communication
4. **EOL date**: Version no longer receives updates

---

## Migration Assistance

### Resources Provided

For every deprecated feature, we provide:

1. **Migration Guide**: Step-by-step documentation for migrating to the new approach

2. **Code Examples**: Before/after code samples showing the migration

3. **Codemods/Scripts**: Automated migration tools when feasible:
   ```bash
   # Example migration script
   python -m aictrlnet.migrate --from v1 --to v2
   ```

4. **Compatibility Shims**: Temporary compatibility layers during transition periods

5. **FAQ Documentation**: Common questions and edge cases addressed

### Deprecation Warnings in Code

AICtrlNet emits clear deprecation warnings:

**Python Warnings:**
```python
import warnings

def deprecated_function():
    warnings.warn(
        "deprecated_function() is deprecated and will be removed in v2.0.0. "
        "Use new_function() instead. "
        "See https://docs.aictrlnet.com/migration/deprecated-function",
        DeprecationWarning,
        stacklevel=2
    )
```

**Log Warnings:**
```
[WARNING] DEPRECATION: The 'legacy_auth' configuration is deprecated.
  - Will be removed in: v2.0.0
  - Use instead: 'oauth2_auth'
  - Migration guide: https://docs.aictrlnet.com/migration/auth
```

**API Response Warnings:**
```json
{
  "warnings": [
    {
      "type": "deprecation",
      "code": "DEPRECATED_ENDPOINT",
      "message": "This endpoint is deprecated",
      "details": {
        "deprecated_in": "1.5.0",
        "removed_in": "2.0.0",
        "alternative": "/api/v2/resource",
        "migration_guide": "https://docs.aictrlnet.com/migration/resource"
      }
    }
  ]
}
```

### Getting Help

If you need assistance with migrations:

1. **Documentation**: Check the migration guides at `docs.aictrlnet.com/migration`
2. **GitHub Discussions**: Ask questions in the community discussions
3. **GitHub Issues**: Report migration problems or edge cases
4. **Office Hours**: Join community office hours for live help (schedule in discussions)

---

## Summary

| Aspect | Policy |
|--------|--------|
| Versioning | Semantic Versioning (MAJOR.MINOR.PATCH) |
| Feature Deprecation Notice | Minimum 6 months or 2 minor versions |
| API Version Support | 12 months after next major version |
| Breaking Changes | Major version bump required |
| Security Patches | Critical: 48 hours, High: 7 days |
| Migration Resources | Guides, examples, scripts, and compatibility shims |

---

*Last updated: January 2026*
*This policy applies to AICtrlNet Community Edition v1.0.0 and later*
