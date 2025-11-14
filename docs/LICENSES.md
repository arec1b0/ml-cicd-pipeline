# License Compliance Documentation

## Project License

This project is licensed under the **MIT License**. See the [LICENSE](../LICENSE) file for details.

## Dependency License Policy

This project maintains strict license compliance to ensure all dependencies are compatible with the MIT license and do not impose unwanted restrictions.

### Approved Licenses

The following licenses are approved for use in this project:

- **MIT License** - Permissive, business-friendly
- **Apache License 2.0** - Permissive with patent protection
- **BSD License (2-Clause, 3-Clause)** - Permissive, simple
- **ISC License** - Functionally equivalent to MIT
- **Python Software Foundation License** - Python-specific permissive license
- **Mozilla Public License 2.0 (MPL-2.0)** - Weak copyleft, file-level
- **LGPL (Lesser GPL)** - Weak copyleft, library usage allowed

### Restricted Licenses

The following licenses are **NOT allowed** without explicit approval:

- **GPL (General Public License)** - Strong copyleft, requires code disclosure
- **AGPL (Affero GPL)** - Network copyleft, requires code disclosure for SaaS
- **SSPL (Server Side Public License)** - Similar to AGPL
- **CC-BY-SA (Creative Commons Share-Alike)** - Copyleft for documentation/media
- **Proprietary/Commercial** - Requires licensing fees or usage restrictions

### License Compliance Workflow

1. **Automated Checking**: The CI pipeline runs `pip-licenses` on every build
2. **License Reports**: Generated reports are uploaded as artifacts (90-day retention)
3. **Copyleft Detection**: Build fails if GPL/AGPL/SSPL licenses are detected
4. **Dependabot Integration**: Automated dependency updates include license verification

### Checking Licenses Locally

To check dependency licenses locally, run:

```bash
# Install pip-licenses
pip install pip-licenses

# View all licenses
pip-licenses

# Generate markdown report
pip-licenses --format=markdown --output-file=licenses-report.md

# Generate JSON report for programmatic access
pip-licenses --format=json --output-file=licenses-report.json

# Check for problematic licenses
pip-licenses | grep -iE "(GPL|AGPL|SSPL)" | grep -v LGPL
```

### Adding New Dependencies

When adding new dependencies:

1. **Check the license** using `pip-licenses` or PyPI
2. **Verify compatibility** with our approved list
3. **Update this document** if a new license type is encountered
4. **Get approval** for any restricted licenses before merging

### Current Dependencies

The project uses the following major dependencies:

#### Core Application
- **FastAPI** (MIT) - Web framework
- **Uvicorn** (BSD-3-Clause) - ASGI server
- **Pydantic** (MIT) - Data validation
- **Python-JSON-Logger** (BSD) - Structured logging

#### Machine Learning
- **scikit-learn** (BSD-3-Clause) - ML algorithms
- **pandas** (BSD-3-Clause) - Data manipulation
- **MLflow** (Apache-2.0) - ML lifecycle management
- **Evidently** (Apache-2.0) - ML monitoring
- **ONNX Runtime** (MIT) - Model inference

#### Observability
- **Prometheus Client** (Apache-2.0) - Metrics collection
- **OpenTelemetry** (Apache-2.0) - Distributed tracing

#### Development Tools
- **pytest** (MIT) - Testing framework
- **ruff** (MIT) - Linting and formatting
- **mypy** (MIT) - Type checking
- **bandit** (Apache-2.0) - Security linting

### License Audit History

| Date | Auditor | Result | Notes |
|------|---------|--------|-------|
| 2025-11-14 | Automated | Pass | Initial audit, all dependencies compliant |

### Contact

For license compliance questions or concerns, please:

1. Open an issue in the repository
2. Contact the project maintainers
3. Review the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines

## References

- [Open Source Initiative - Licenses](https://opensource.org/licenses)
- [Choose a License](https://choosealicense.com/)
- [TLDRLegal - Software Licenses Explained](https://www.tldrlegal.com/)
- [SPDX License List](https://spdx.org/licenses/)

---

*Last updated: 2025-11-14*
*This document should be reviewed quarterly or when significant dependency changes occur.*
