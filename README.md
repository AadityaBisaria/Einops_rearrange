# Einops_rearrange

The implementation is built around four core components that work together in a transformation pipeline. The `PatternParser` converts string patterns into a tree of `PatternNode` objects, which represent the hierarchical structure of the pattern (including individual axes, groups, and ellipsis). The `AxisOperation` class represents individual transformations (like splitting, merging, or transposing axes) and is categorized by the `OperationType` enum. These components work together to first parse the pattern, then infer the necessary transformations, and finally apply them sequentially to transform the tensor while maintaining shape consistency and proper axis ordering.

## Key Features

- Type Safety: Comprehensive type hints throughout the codebase  
- Flexible Pattern Syntax: Supports complex patterns with groups and ellipsis  
- Dimension Inference: Can infer unknown dimensions based on known ones  
- Error Handling: Detailed error messages for invalid patterns or operations  
- Debugging Support: Extensive logging for operation tracking  

## Key Implementation Aspects

- Pattern parsing using recursive descent  
- Operation inference through pattern comparison  
- Shape tracking and validation  
- Dimension size inference and validation  
- Error handling and debugging support  

## Order of Operations

The transformations are applied in a specific order to ensure correct results:

1. MERGE operations first (combining dimensions)  
2. SPLIT operations second (breaking down dimensions)  
3. REPEAT operations third (adding new dimensions)  
4. TRANSPOSE operations last (reordering dimensions)  

This order is crucial because:

- Merging must happen before splitting to avoid conflicts  
- Repeating needs to happen after splitting to handle new dimensions  
- Transposing is done last to ensure all dimensions are in their final form

  ## Usage Example

```python
import numpy as np

# Example tensor
tensor = np.random.rand(32, 64, 64, 3)  # (batch, height, width, channels)

# Rearrange using pattern
result = rearrange(tensor, "b h w c -> b c h w")
# Result shape: (32, 3, 64, 64)
```


