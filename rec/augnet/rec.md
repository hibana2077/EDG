# Augnet - experimental record

| Augnet architecture | Highest accuracy| w/o Augmentation | Augnet Loss | Notes |
|---------------------|------------------|------------------|-------------|-------|
| Kornia Normalization | 0.4961           | 0.4625           |   0    | baseline model: tiny_vit_21m_224 |
| Kornia Norm + Rand Sharp | 0.4102           | 0.4625           |  0.7762     | baseline model: tiny_vit_21m_224 |
| Kornia Norm + Rand Const | 0.3750           | 0.4625           |  0.0373     | baseline model: tiny_vit_21m_224 |

## image

### Kornia Normalization

![Kornia Normalization](./norm.png)

### Kornia Norm + Rand Sharp

![Kornia Norm + Rand Sharp](./norm_rs.png)

### Kornia Norm + Rand Const

![Kornia Norm + Rand Const](./norm_rc.png)