def GReg_loss(matrix_data, epsilon=1e-8):
    m = matrix_data.size(1)  # latent dim
    abs_cube = torch.sum(torch.abs(matrix_data) ** 3, dim=1) / m
    square_mean = torch.sum(matrix_data ** 2, dim=1) / m
    denom = square_mean.clamp(min=epsilon) ** (3/2)
    result = abs_cube / (denom + epsilon)
    l2_reg = torch.mean(torch.sum(matrix_data ** 2, dim=1)) 
    return torch.mean(result) + 1e-2 * l2_reg
