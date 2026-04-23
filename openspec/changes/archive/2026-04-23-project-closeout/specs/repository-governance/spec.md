## ADDED Requirements

### Requirement: OpenSpec is the only active governance source
The repository SHALL use `openspec/` as the only active requirements and change-management authority for future work.

#### Scenario: Contributor inspects governance files
- **WHEN** a contributor reviews repository governance documentation
- **THEN** the contributor MUST be directed to `openspec/specs/` and `openspec/changes/` instead of a parallel legacy specification system

### Requirement: Closeout scope is explicitly bounded
The repository SHALL represent closeout stabilization as the active project priority and MUST distinguish deferred feature work from closeout-critical work.

#### Scenario: Deferred feature proposals exist
- **WHEN** unfinished feature-oriented changes remain in the repository
- **THEN** those changes MUST be marked or described as deferred backlog rather than implied release-blocking work for the closeout cycle

### Requirement: Repository structure remains intentional
The repository SHALL keep only directories and governance files that have an active purpose in the final closeout-era structure.

#### Scenario: Legacy governance structure is superseded
- **WHEN** a directory or file exists solely to preserve an obsolete planning workflow
- **THEN** that directory or file MUST be removed or reduced so it no longer presents itself as an active source of truth

### Requirement: Verification distinguishes setup failures from product defects
The repository SHALL treat environment preparation failures separately from code defects during closeout verification.

#### Scenario: Validation fails before tests execute
- **WHEN** a validation command fails because required local dependencies are missing
- **THEN** the failure MUST be recorded as an environment baseline issue and MUST NOT be treated as a product defect until the documented environment is prepared
