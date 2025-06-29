# Transformation Analysis Tests

This directory contains tests for the GPT-2 context transformation analysis framework.

## Test Structure

- `conftest.py` - Shared test fixtures and configuration
- `test_output_schema.py` - Tests for the unified output schema
- `test_data_loader.py` - Tests for the TransformationDataLoader
- `test_base_transformation_analysis.py` - Tests for the base analysis class

## Running Tests

### Install test dependencies:
```bash
cd experiments/gpt2_pronouns/transformation_analysis
pip install -r requirements-test.txt
```

### Run all tests:
```bash
# From transformation_analysis directory
python run_tests.py

# Or use pytest directly
pytest -v
```

### Run specific test files:
```bash
pytest tests/test_output_schema.py -v
```

### Run tests by marker:
```bash
# Only unit tests
pytest -m unit

# Only integration tests  
pytest -m integration
```

### Generate coverage report:
```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html to view report
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Use mocked dependencies
- Should be fast and deterministic

### Integration Tests (`@pytest.mark.integration`)
- Test interaction between components
- May use real file I/O
- Test complete pipelines

### Slow Tests (`@pytest.mark.slow`)
- Tests that take significant time
- Run separately: `pytest -m slow`

## Writing New Tests

1. Follow existing patterns in test files
2. Use appropriate fixtures from `conftest.py`
3. Mark tests with appropriate categories
4. Ensure tests are independent and can run in any order
5. Mock external dependencies (file I/O, API calls)

## Test Coverage Goals

- Minimum 80% coverage for all modules
- 100% coverage for critical paths (data loading, validation)
- All error cases should be tested