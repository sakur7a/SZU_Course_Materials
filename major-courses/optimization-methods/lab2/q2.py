import numpy as np
import scipy.io
import scipy.linalg
import os

def invert_via_qr(B):
    """
    使用QR分解方法判断矩阵B的可逆性并求解其逆。
    """
    print("\n--- 使用QR分解方法判断可逆性与求逆 ---")
    
    # 3.5.1 可逆性判断
    # 1. 对B做QR分解
    print("1. 正在对矩阵B进行QR分解...")
    try:
        # 使用scipy.linalg.qr以获得更好的控制
        Q, R = scipy.linalg.qr(B)
        print("   分解完成。")
    except np.linalg.LinAlgError as e:
        print(f"   QR分解失败: {e}")
        return None, False

    # 2. 检查R的对角线元素
    print("\n2. 正在判断可逆性...")
    n = B.shape[0]
    # 取R对角线元素的绝对值的最小值
    min_diag_abs = np.min(np.abs(np.diag(R)))
    print(f"   R矩阵的最小对角线元素绝对值为: {min_diag_abs:.4e}")

    # 设置一个阈值来判断奇异性
    invertibility_threshold = 1e-6
    
    if min_diag_abs <= invertibility_threshold:
        print(f"   结论: 矩阵B不可逆 (存在 |R_ii| <= {invertibility_threshold})。")
        return None, False
    else:
        print(f"   结论: 矩阵B可逆 (所有 |R_ii| > {invertibility_threshold})。")

    # 3.5.2 逆矩阵求解
    print("\n3. 正在求解逆矩阵...")
    # 1. 求解 R*X = I, 得 X = R_inv
    print("   通过求解 RX = I 得到 R_inv...")
    identity = np.eye(n)
    # 使用scipy专门为三角矩阵设计的求解器，高效且稳定
    R_inv = scipy.linalg.solve_triangular(R, identity)

    # 2. 计算 B_inv = R_inv * Q.T
    print("   通过 B_inv = R_inv @ Q.T 计算 B_inv...")
    B_inv = R_inv @ Q.T
    print("   逆矩阵求解完成。")

    # 3. 验证
    print("\n4. 正在验证结果...")
    verification_error = np.linalg.norm(B @ B_inv - identity, 'fro')
    print(f"   验证误差 ||B * B_inv - I||_fro: {verification_error:.4e}")
    if verification_error < 1e-6: # 使用一个合理的误差阈值
        print("   验证通过，结果有效。")
        return B_inv, True
    else:
        print("   警告：验证误差较大，结果可能不准确。")
        return B_inv, False


def main(matrix_file="MatrixB.mat", variable_name='B'):
    """
    主函数，读取矩阵并执行通过QR分解求逆的流程。
    """
    # 1. 读取.mat文件
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        matrix_path = os.path.join(script_dir, matrix_file)
        mat_data = scipy.io.loadmat(matrix_path)
        B = mat_data[variable_name]
        print(f"成功读取矩阵 {variable_name}，其形状为: {B.shape}")
    except FileNotFoundError:
        print(f"错误: 未在路径 '{matrix_path}' 中找到 '{matrix_file}' 文件。")
        return
    except KeyError:
        print(f"错误: '{matrix_file}' 文件中没有名为 '{variable_name}' 的变量。")
        return

    # 2. 判断是否为方阵
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        print(f"\n结论: 矩阵 {variable_name} 不是方阵，因此不可逆。")
        return

    # 3. 调用QR方法进行求逆
    B_inv, is_successful = invert_via_qr(B)

    # 4. 如果成功，则保存和展示
    if is_successful and B_inv is not None:
        # 保存为.mat文件
        output_mat_path = os.path.join(script_dir, 'MatrixB_inverse.mat')
        scipy.io.savemat(output_mat_path, {'B_inv': B_inv})
        print(f"\n逆矩阵已成功保存到: {output_mat_path}")


if __name__ == '__main__':
    main()

