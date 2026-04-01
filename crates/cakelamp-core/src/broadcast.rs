/// Compute the broadcast shape of two shapes, following NumPy/PyTorch rules.
/// Returns None if the shapes are not broadcastable.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let da = if i < max_ndim - a.len() { 1 } else { a[i - (max_ndim - a.len())] };
        let db = if i < max_ndim - b.len() { 1 } else { b[i - (max_ndim - b.len())] };

        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return None;
        }
    }

    Some(result)
}

/// Compute strides for broadcasting `shape` to `target_shape`.
/// Dimensions that were size 1 (broadcast) get stride 0.
pub fn broadcast_strides(shape: &[usize], strides: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let ndim = target_shape.len();
    let offset = ndim - shape.len();
    let mut result = vec![0usize; ndim];

    for i in 0..ndim {
        if i < offset {
            result[i] = 0; // prepended dimension
        } else {
            let orig_idx = i - offset;
            if shape[orig_idx] == target_shape[i] {
                result[i] = strides[orig_idx];
            } else {
                // shape[orig_idx] == 1, broadcast
                result[i] = 0;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape_same() {
        assert_eq!(broadcast_shape(&[3, 4], &[3, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shape_scalar() {
        assert_eq!(broadcast_shape(&[3, 4], &[1]), Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shape_diff_ndim() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[5, 3, 1], &[4]), Some(vec![5, 3, 4]));
    }

    #[test]
    fn test_broadcast_shape_incompatible() {
        assert_eq!(broadcast_shape(&[3, 4], &[3, 5]), None);
    }
}
