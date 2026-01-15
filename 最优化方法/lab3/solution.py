import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    读取 .mat 文件中的 A 和 b
    """
    data = scipy.io.loadmat(filepath)
    # 假设 mat 文件中的变量名为 'A' 和 'b'
    # 如果变量名不同，需要根据实际情况修改
    # 这里我们先打印一下 keys 确认
    print(f"Keys in {filepath}: {data.keys()}")
    
    # 直接获取 A 和 b
    A = data['A']
    b = data['b']
                
    # 确保 b 是列向量
    if b.ndim == 1:
        b = b.reshape(-1, 1)
        
    return A, b

def solve_exact(A, b):
    """
    使用正规方程求解准确解: x = (A^T A)^(-1) A^T b
    
    原理说明：
    虽然矩阵 A (50x40) 不是方阵且不可逆，但我们之前验证了 A 具有满列秩 (Rank=40)。
    这意味着方阵 A^T A (40x40) 是可逆的（非奇异）。
    因此，我们可以通过求解线性方程组 (A^T A) x = A^T b 来得到唯一的准确解。
    """
    ATA = A.T @ A
    ATb = A.T @ b
    # 使用 np.linalg.solve 求解线性方程组 ATA * x = ATb
    # 这在数学上等价于 x = inv(ATA) * ATb，但数值计算上更稳定且更准确
    x_exact = np.linalg.solve(ATA, ATb)
    return x_exact

def gradient_descent(A, b, learning_rate=0.001, max_iters=10000, tol=1e-6):
    """
    使用梯度下降法求解近似解
    """
    m, n = A.shape
    x = np.zeros((n, 1)) # 初始化 x
    
    history = []
    
    for k in range(max_iters):
        # 计算梯度: grad = 2 * A^T * (Ax - b)
        residual = A @ x - b
        grad = 2 * A.T @ residual
        
        x_new = x - learning_rate * grad
        
        # 计算相对误差作为停止条件
        norm_x = np.linalg.norm(x)
        diff = np.linalg.norm(x_new - x)
        
        if norm_x > 0:
            relative_change = diff / norm_x
        else:
            relative_change = diff # 避免除以零
            
        history.append(np.linalg.norm(residual)**2) # 记录目标函数值
        
        if k % 10000 == 0:
             print(f"Iteration {k}: Loss = {history[-1]:.6f}, Relative Change = {relative_change:.2e}")

        if relative_change < tol:
            print(f"Converged at iteration {k}, relative change: {relative_change:.2e}")
            break
            
        x = x_new
        
    return x, history

def main():
    filepath = 'Matrix_A_b.mat'
    try:
        A, b = load_data(filepath)
        print(f"Loaded A shape: {A.shape}, b shape: {b.shape}")
        
        # 0. 验证矩阵 A 的性质
        print("\n--- Verifying Matrix Properties ---")
        m, n = A.shape
        if m != n:
            print(f"Matrix A is {m}x{n}, not square, so it is NOT invertible.")
        else:
            det_A = np.linalg.det(A)
            print(f"Matrix A is square. Determinant: {det_A}")
            if np.abs(det_A) < 1e-10:
                print("Matrix A is singular (not invertible).")
            else:
                print("Matrix A is invertible.")

        # 检查 A 的秩
        rank_A = np.linalg.matrix_rank(A)
        print(f"Rank of A: {rank_A}")
        
        if rank_A == n:
            print(f"Matrix A has full column rank ({n}). Therefore, A^T A is invertible.")
            print("The least squares problem has a unique solution.")
        else:
            print(f"Matrix A does not have full column rank. A^T A is singular.")
            
        # 检查 A^T A 的条件数
        cond_ATA = np.linalg.cond(A.T @ A)
        print(f"Condition number of A^T A: {cond_ATA:.2e}")
        print("-----------------------------------\n")
        
        # 1. 准确解
        print("Calculating exact solution...")
        x_exact = solve_exact(A, b)
        print("Exact solution x (first 10 elements):")
        print(x_exact[:10].flatten())
        print("Exact solution x (full vector):")
        print(x_exact.flatten())
        loss_exact = np.linalg.norm(A @ x_exact - b)**2
        print(f"Exact solution loss: {loss_exact:.6f}")

        # 保存准确解到 .mat 文件
        scipy.io.savemat('x_exact.mat', {'x_exact': x_exact})
        print("Exact solution saved to 'x_exact.mat'")

        # (3) 解的有效性验证
        print("\n--- Validating Exact Solution ---")
        # 1. 验证梯度是否为 0: grad f(x) = 2A^T(Ax - b)
        grad_exact = 2 * A.T @ (A @ x_exact - b)
        grad_norm = np.linalg.norm(grad_exact)
        print(f"Norm of gradient at x_exact: {grad_norm:.6e}")
        
        if grad_norm < 1e-10:
            print("Gradient validation PASSED (norm < 1e-10).")
        else:
            print("Gradient validation FAILED.")

        # 2. 正交性检验: A^T * r_exact approx 0, where r_exact = b - Ax
        r_exact = b - A @ x_exact
        ortho_check = A.T @ r_exact
        ortho_norm = np.linalg.norm(ortho_check)
        print(f"Norm of A^T * r_exact: {ortho_norm:.6e}")
        
        if ortho_norm < 1e-10:
             print("Orthogonality check PASSED (norm < 1e-10).")
        else:
             print("Orthogonality check FAILED.")
        print("---------------------------------\n")
        
        # (3) 解的有效性验证
        print("\n--- Validating Exact Solution ---")
        # 1. 验证梯度是否为 0: grad f(x) = 2A^T(Ax - b)
        grad_exact = 2 * A.T @ (A @ x_exact - b)
        grad_norm = np.linalg.norm(grad_exact)
        print(f"Norm of gradient at x_exact: {grad_norm:.6e}")
        
        if grad_norm < 1e-10:
            print("Gradient validation PASSED (norm < 1e-10).")
        else:
            print("Gradient validation FAILED.")

        # 2. 正交性检验: A^T * r_exact approx 0, where r_exact = b - Ax
        r_exact = b - A @ x_exact
        ortho_check = A.T @ r_exact
        ortho_norm = np.linalg.norm(ortho_check)
        print(f"Norm of A^T * r_exact: {ortho_norm:.6e}")
        
        if ortho_norm < 1e-10:
             print("Orthogonality check PASSED (norm < 1e-10).")
        else:
             print("Orthogonality check FAILED.")
        print("---------------------------------\n")
        
        # 2. 近似解 (梯度下降)
        # 理论上最佳步长 < 2/L, 其中 L 是 Hessian 矩阵 2*A^T*A 的最大特征值
        L = np.linalg.norm(A.T @ A, 2) * 2
        lr = 1.0 / L 
        print(f"Lipschitz constant L: {L:.6f}")
        print(f"Theoretical optimal learning rate (alpha_opt = 1/L): {lr:.6f}")
        
        print("Calculating approximate solution via Gradient Descent...")
        # 增加最大迭代次数以提高精度
        x_approx, history = gradient_descent(A, b, learning_rate=lr, max_iters=200000, tol=1e-8)
        loss_approx = np.linalg.norm(A @ x_approx - b)**2
        print(f"Approximate solution loss: {loss_approx:.6f}")

        # 保存近似解到 .mat 文件
        scipy.io.savemat('x_approx.mat', {'x_approx': x_approx})
        print("Approximate solution saved to 'x_approx.mat'")
        
        # 3. 误差分析与收敛性分析
        error = np.linalg.norm(x_approx - x_exact)
        print(f"L2 Error between exact and approx solution: {error:.6e}")
        
        # 相对误差
        rel_error = error / np.linalg.norm(x_exact)
        print(f"Relative Error: {rel_error:.6e}")

        # 绘制收敛曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history, label=f'LR={lr:.2e}')
        plt.axhline(y=loss_exact, color='r', linestyle='--', label='Exact Solution Loss')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Convergence Analysis: Loss vs Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig('convergence_analysis.png')
        print("Convergence plot saved as 'convergence_analysis.png'")

        # 4. 参数敏感性分析
        print("\n--- Parameter Sensitivity Analysis ---")
        # 选取一组不同的学习率进行对比
        learning_rates = [lr * 0.1, lr * 0.5, lr, lr * 1.5, lr * 2.0]
        plt.figure(figsize=(10, 6))
        
        for current_lr in learning_rates:
            print(f"Testing learning rate: {current_lr:.2e}")
            # 为了绘图清晰，这里减少迭代次数或保持一致
            _, hist = gradient_descent(A, b, learning_rate=current_lr, max_iters=2000, tol=1e-6)
            plt.plot(hist, label=f'LR={current_lr:.2e}')
            
        plt.axhline(y=loss_exact, color='r', linestyle='--', label='Exact Solution Loss')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Parameter Sensitivity: Effect of Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig('parameter_sensitivity.png')
        print("Sensitivity plot saved as 'parameter_sensitivity.png'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
