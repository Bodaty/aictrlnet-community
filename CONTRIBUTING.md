# Contributing to AICtrlNet

Thank you for your interest in contributing to AICtrlNet! This document provides
guidelines for contributing to the Community Edition.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/aictrlnet-community.git
   cd aictrlnet-community
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_your_feature.py
```

### 4. Lint Your Code

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking (optional)
mypy src/
```

### 5. Commit Your Changes

Write clear, concise commit messages:

```bash
git commit -m "feat: add new adapter for XYZ service"
git commit -m "fix: resolve authentication timeout issue"
git commit -m "docs: update API documentation for tasks endpoint"
```

Commit message prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## Code Style Guidelines

### Python

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 120 characters
- Use descriptive variable names

### API Endpoints

- Use RESTful conventions
- Include proper error handling
- Add Pydantic schemas for request/response validation
- Document endpoints with docstrings (shows in Swagger)

### Tests

- Write unit tests for new functionality
- Aim for 80%+ code coverage
- Use pytest fixtures for setup
- Mock external services

## Pull Request Guidelines

1. **One feature per PR** - Keep PRs focused and reviewable
2. **Update tests** - All new code should have tests
3. **Update docs** - Document new features or API changes
4. **Pass CI** - All tests and linting must pass
5. **Respond to feedback** - Be responsive to review comments

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Feature Requests

When requesting features:
- Describe the use case
- Explain why it benefits the community
- Consider if it belongs in Community vs Business/Enterprise edition

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

- Open a [GitHub Discussion](https://github.com/Bodaty/aictrlnet-community/discussions)
- Email: team@aictrlnet.com

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
