# Code Enhancement: data-science-mcp

> Automated code enhancement review for data-science-mcp. Covers 17 analysis domains.

## User Stories

- As a **developer**, I want to **address Project Analysis findings (grade: D, score: 69)**, so that **improve project project analysis from D to at least B (80+)**.
- As a **developer**, I want to **address Codebase Optimization findings (grade: D, score: 63)**, so that **improve project codebase optimization from D to at least B (80+)**.
- As a **developer**, I want to **address Concept Traceability findings (grade: F, score: 24)**, so that **improve project concept traceability from F to at least B (80+)**.
- As a **developer**, I want to **address Test Execution findings (grade: F, score: 25)**, so that **improve project test execution from F to at least B (80+)**.
- As a **developer**, I want to **address Changelog Audit findings (grade: C, score: 75)**, so that **improve project changelog audit from C to at least B (80+)**.
- As a **developer**, I want to **address analyze_xdg_kg findings (grade: F, score: 0)**, so that **improve project analyze_xdg_kg from F to at least B (80+)**.

## Functional Requirements

- **FR-001**: Minor update: agent-utilities 0.2.40 (installed) -> 0.16.0
- **FR-002**: Minor update: scikit-learn 1.5.0 (constraint — not installed) -> 1.8.0
- **FR-003**: MAJOR update: pandas 2.0.0 (constraint — not installed) -> 3.0.3
- **FR-004**: 2 functions exceed 200 lines (actionable refactoring targets): register_interpretability_tools (212L), register_interpretability_tools (212L)
- **FR-005**: Monolithic: mcp_server.py (663L) — 2 functions with high complexity (worst: register_interpretability_tools at 212L, CC=24); Low cohesion: 11 distinct concepts in one file
- **FR-006**: Test suite lacks intent diversity (only one type)
- **FR-007**: 12 potential doc-test drift items
- **FR-008**: README.md missing sections: usage|quick start
- **FR-009**: 2 broken internal links in README.md
- **FR-010**: README missing: Has a Table of Contents
- **FR-011**: README missing: Has usage examples with code blocks
- **FR-012**: SRP: 2 modules exceed 500 lines (god modules)
- **FR-013**: No discernible layer architecture (no domain/service/adapter separation)
- **FR-014**: Low traceability ratio: 0% concepts fully traced
- **FR-015**: 13 orphaned concepts (only in one source)
- **FR-016**: 28 test functions missing concept markers
- **FR-017**: 47 significant functions (>10 lines) missing concept markers in docstrings
- **FR-018**: Total lint findings: 0 (high/error: 0, medium/warning: 0, low: 0)
- **FR-019**: 1 hook(s) may be outdated: ruff-pre-commit
- **FR-020**: CHANGELOG.md exists but could not be parsed — check format compliance
- **FR-021**: No changelog entries within the last 30 days
- **FR-022**: keepachangelog not installed — pip install 'universal-skills[code-enhancer]'
- **FR-023**: 1 test files exceed 500 lines — split into focused modules
- **FR-024**: No @pytest.mark.parametrize usage — consider data-driven tests
- **FR-025**: 10 tests have >5 assertions — consider splitting (single responsibility)
- **FR-026**: 1 tests exceed 100 lines — likely doing too much per test
- **FR-027**: Partial env var documentation: 53% coverage
- **FR-028**: Undocumented env vars: AUTH_TYPE, TLS_PROFILES_REF, TLS_PROFILE, EUNOMIA_POLICY_FILE, EUNOMIA_TYPE, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_PROTOCOL, OTEL_EXPORTER_OTLP_PUBLIC_KEY, OTEL_EXPORTER_OTLP_SECRET_KEY
- **FR-029**: 2 Python env vars not in .env.example: TLS_PROFILES_REF, TLS_PROFILE
- **FR-030**: Analysis error: No module named 'agent_utilities.knowledge_graph'

## Success Criteria

- Overall GPA: 2.53 → 3.0
- Domains at B or above: 11 → 17
- Actionable findings: 30 → 0
