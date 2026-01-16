# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in AICtrlNet,
please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please email: **security@bodaty.com**

Include the following information:
- Type of vulnerability (e.g., SQL injection, XSS, authentication bypass)
- Location of the affected code (file path, line number if known)
- Steps to reproduce the vulnerability
- Potential impact assessment
- Any suggested fixes (optional)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Assessment**: We will assess the vulnerability within 5 business days
3. **Updates**: We will keep you informed of our progress
4. **Resolution**: We aim to resolve critical issues within 30 days
5. **Credit**: With your permission, we will credit you in the release notes

### Security Best Practices for Deployment

When deploying AICtrlNet, follow these security guidelines:

#### Authentication
- Use strong, unique JWT secrets
- Implement token rotation policies

#### Network Security
- Deploy behind a reverse proxy (nginx, Traefik)
- Use HTTPS/TLS for all communications
- Restrict database access to application servers only

#### Database Security
- Use strong database passwords
- Enable connection encryption
- Implement regular backups

#### Container Security
- Keep base images updated
- Scan images for vulnerabilities
- Run containers as non-root users

## Security Features

AICtrlNet Community Edition includes:

- **JWT Authentication**: Stateless token-based authentication
- **Rate Limiting**: Protection against brute force and DoS attacks
- **Input Validation**: Sanitization of user inputs
- **Audit Logging**: Comprehensive logging for compliance

## Disclosure Policy

We follow a coordinated disclosure policy:
- Vulnerabilities are disclosed publicly only after a fix is available
- We coordinate with reporters on disclosure timing
- Critical vulnerabilities may warrant expedited disclosure
