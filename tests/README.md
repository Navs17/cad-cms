# Tests

Pytest unit tests for the memory math (Phase 3), e.g.:

- fusing two known Gaussians gives the expected moments
- shrinkage keeps the covariance invertible for small sample counts
- EMA updates converge to a stationary mean

Run with:

```bash
pytest tests/
```
