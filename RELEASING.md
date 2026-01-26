# AICtrlNet Community Edition - Release Process

This document describes the release process for AICtrlNet Community Edition.

## Versioning Strategy

We follow [Semantic Versioning](https://semver.org/) (SemVer):

```
MAJOR.MINOR.PATCH
```

| Component | When to Increment | Example |
|-----------|-------------------|---------|
| **MAJOR** | Breaking API changes, major rewrites | 1.0.0 → 2.0.0 |
| **MINOR** | New features, backward-compatible | 1.0.0 → 1.1.0 |
| **PATCH** | Bug fixes, backward-compatible | 1.0.0 → 1.0.1 |

### Pre-release Versions

- Alpha: `1.0.0-alpha.1`
- Beta: `1.0.0-beta.1`
- Release Candidate: `1.0.0-rc.1`

### Current Version

The current version is defined in:
- `editions/community/setup.py` → `version="X.Y.Z"`
- `editions/community/src/core/config.py` → `VERSION: str = "X.Y.Z"`

## Release Checklist

### Pre-Release

- [ ] All tests passing (`make test`)
- [ ] No critical bugs in issue tracker
- [ ] CHANGELOG.md updated with release notes
- [ ] Version numbers updated in all files
- [ ] Documentation up to date

### Version Bump

1. **Update version in setup.py:**
   ```python
   setup(
       name="aictrlnet",
       version="X.Y.Z",  # Update this
       ...
   )
   ```

2. **Update version in config.py:**
   ```python
   VERSION: str = "X.Y.Z"  # Update this
   ```

3. **Update CHANGELOG.md:**
   ```markdown
   ## [X.Y.Z] - YYYY-MM-DD

   ### Added
   - New feature description

   ### Changed
   - Changed feature description

   ### Fixed
   - Bug fix description
   ```

### Create Release

1. **Commit version changes:**
   ```bash
   git add setup.py src/core/config.py CHANGELOG.md
   git commit -m "chore: bump version to X.Y.Z"
   ```

2. **Create and push tag:**
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main
   git push origin vX.Y.Z
   ```

3. **GitHub Release triggers automatic:**
   - Docker image build and push to Docker Hub/GHCR
   - PyPI package build and upload (if configured)

### Manual Release (if needed)

**Docker Images:**
```bash
VERSION=X.Y.Z make release
# This runs: build-dist → verify-dist → push
```

**PyPI Package:**
```bash
cd editions/community
python -m build
twine upload dist/*
```

## Release Artifacts

Each release produces:

| Artifact | Location | Command |
|----------|----------|---------|
| Docker Image | `bodaty/aictrlnet-community:X.Y.Z` | `docker pull bodaty/aictrlnet-community:X.Y.Z` |
| Docker Image | `ghcr.io/bodaty/aictrlnet-community:X.Y.Z` | `docker pull ghcr.io/bodaty/aictrlnet-community:X.Y.Z` |
| PyPI Package | pypi.org/project/aictrlnet | `pip install aictrlnet==X.Y.Z` |
| Source | GitHub Release | Download from Releases page |

## Hotfix Process

For urgent bug fixes:

1. **Create hotfix branch from tag:**
   ```bash
   git checkout -b hotfix/X.Y.Z vX.Y.Z
   ```

2. **Apply fix and bump patch version:**
   ```bash
   # Make fix
   # Update version to X.Y.(Z+1)
   git commit -m "fix: critical bug description"
   ```

3. **Create new tag and release:**
   ```bash
   git tag -a vX.Y.(Z+1) -m "Hotfix release"
   git push origin vX.Y.(Z+1)
   ```

4. **Merge back to main:**
   ```bash
   git checkout main
   git merge hotfix/X.Y.Z
   ```

## Release Schedule

| Type | Frequency | Notes |
|------|-----------|-------|
| Major | As needed | Breaking changes, planned well in advance |
| Minor | Monthly | New features, announced 1 week ahead |
| Patch | As needed | Bug fixes, can be immediate for critical issues |

## Support Policy

| Version | Support Status |
|---------|----------------|
| Latest (X.Y.Z) | Full support |
| Previous minor (X.Y-1.Z) | Security fixes only |
| Older | No support |

## Rollback Procedure

If a release needs to be rolled back:

1. **Docker:**
   ```bash
   docker pull bodaty/aictrlnet-community:PREVIOUS_VERSION
   docker-compose down
   # Update docker-compose.yml to use previous version
   docker-compose up -d
   ```

2. **Mark release as broken on GitHub:**
   - Edit the release
   - Add warning about known issues
   - Point to previous stable version

3. **Communicate:**
   - Post in GitHub Discussions
   - Update status page (if applicable)

## Automation

The release process is automated via GitHub Actions:

- **On tag push (`v*`):** `.github/workflows/docker-publish.yml`
  - Builds multi-arch Docker images
  - Pushes to Docker Hub and GHCR
  - Updates GitHub Release with Docker info

- **On PR merge to main:** `.github/workflows/ci.yml`
  - Runs all tests
  - Validates build

## Contact

Release questions: bobby@bodaty.com
