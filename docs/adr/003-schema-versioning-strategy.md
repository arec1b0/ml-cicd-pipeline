# ADR 003: Schema Versioning for Data and Model Contracts

## Status
Accepted

## Context

As ML models evolve, input features, output formats, and data schemas change. Without explicit versioning, production services may face:
- Incompatible input/output contracts between training and serving
- Silent failures when features are renamed or removed
- Difficulty debugging which schema version a model expects
- Integration issues when multiple model versions coexist (canary deployments)

## Decision

We implement **explicit schema versioning** for both model inputs (features) and outputs (predictions) using semantic versioning coupled with MLflow model metadata.

## Rationale

1. **Explicit Contracts**: Each model version explicitly declares its expected input schema and output format.

2. **Semantic Versioning**: 
   - MAJOR: Breaking changes to input/output schema
   - MINOR: New optional features added
   - PATCH: Bug fixes, no schema changes

3. **MLflow Integration**: Store schema versions in model metadata tags:
   ```
   schema_version: "1.0.0"
   feature_names: ["feature_1", "feature_2", "feature_3", ...]
   feature_types: ["float64", "int32", "category", ...]
   output_schema: "predictions: List[int]"
   ```

4. **Validation at Runtime**: The API validates inputs against the model's declared schema, preventing silent failures.

5. **Documentation**: Schema versions create a clear audit trail for feature engineering changes over time.

## Schema Definition Format

### Training Data Schema
```yaml
# configs/schema/v1.0.0.yaml
version: "1.0.0"
created_date: "2024-01-15"
model_version: "iris-random-forest:1"

features:
  - name: "sepal_length"
    type: "float64"
    min: 4.3
    max: 7.9
    description: "Iris sepal length in cm"
  
  - name: "sepal_width"
    type: "float64"
    min: 2.0
    max: 4.4
    description: "Iris sepal width in cm"

output:
  - name: "prediction"
    type: "int32"
    values: [0, 1, 2]
    description: "Iris species class"
```

### Validation Implementation
- Use **Pydantic** models with explicit version constraints
- Implement schema validators in `src/data/validators/schema.py`
- Reject requests with incompatible schemas at API boundary

## Alternatives Considered

- **Implicit Schema Discovery**: Infer schema from data (error-prone, no explicit contracts)
- **Single Global Schema**: Doesn't support multiple model versions or gradual rollouts
- **No Versioning**: Ad-hoc approach, difficult to maintain and debug

## Consequences

- **Positive**:
  - Clear input/output contracts prevent surprises
  - Easy debugging of schema mismatches
  - Support for gradual schema evolution
  - Canary deployments with different model versions
  - Compliance and audit trails

- **Negative**:
  - Additional configuration overhead
  - Need to update schema on each model change
  - More validation logic in API layer

## Migration Path

1. **Phase 1**: Add schema versioning to new models
2. **Phase 2**: Retrofit existing models with schema versions
3. **Phase 3**: Enforce schema validation in all deployments

## Implementation Roadmap

- [ ] Create schema validation library in `src/data/validators/`
- [ ] Add schema version to model metadata during training
- [ ] Update API to validate against model schema
- [ ] Create schema evolution documentation
- [ ] Add schema versioning to monitoring/alerting

## See Also

- `docs/MODEL_LIFECYCLE.md` - Model versioning and promotion
- `src/data/validators/` - Data validation implementation
- `src/app/api/predict.py` - API input validation

