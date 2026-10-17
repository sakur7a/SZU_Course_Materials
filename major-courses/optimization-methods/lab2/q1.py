import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

def classic_gram_schmidt(A):
    """
    使用经典的Gram-Schmidt算法对矩阵A进行QR分解。

    参数:
    A (np.ndarray): 输入的 m x n 矩阵。

    返回:
    Q (np.ndarray): m x n 的正交矩阵。
    R (np.ndarray): n x n 的上三角矩阵。
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i].T, A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10: # 避免除以零
            raise np.linalg.LinAlgError("矩阵的列向量线性相关")
            
        R[j, j] = norm_v
        Q[:, j] = v / R[j, j]
        
    return Q, R

def modified_gram_schmidt(A):
    """
    使用修正的Gram-Schmidt算法对矩阵A进行QR分解。
    这个版本在数值上比经典GS更稳定。
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            # 关键区别: 使用更新后的v来计算点积
            R[i, j] = np.dot(Q[:, i].T, v)
            v = v - R[i, j] * Q[:, i]
        
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            raise np.linalg.LinAlgError("矩阵的列向量线性相关")
            
        R[j, j] = norm_v
        Q[:, j] = v / R[j, j]
        
    return Q, R

def householder_qr(A):
    """
    使用Numpy内置的Householder QR分解。
    np.linalg.qr() 默认使用Householder变换，是数值稳定的。
    """
    return np.linalg.qr(A)

def main():
    # 设定误差阈值
    F_NORM_THRESHOLD = 1e-6

    # 1. 读取MatrixA.mat文件中的矩阵A
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        matrix_a_path = os.path.join(script_dir, 'MatrixA.mat')
        mat_data = scipy.io.loadmat(matrix_a_path)
        A = mat_data['A']
        print(f"成功读取矩阵 A，其形状为: {A.shape}")
    except FileNotFoundError:
        print(f"错误: 未在路径 '{matrix_a_path}' 中找到 'MatrixA.mat' 文件。")
        return
    except KeyError:
        print("错误: 'MatrixA.mat' 文件中没有名为 'A' 的变量。")
        return

    # 计算矩阵A的条件数
    cond_A = np.linalg.cond(A)
    print(f"\n矩阵 A 的条件数: {cond_A:.4e}")
    if cond_A > 1e10:
        print("条件数非常大，表明矩阵A是病态的，这可能导致经典GS算法不稳定。")

    results = []
    
    # --- 方法一: 经典 Gram-Schmidt ---
    print("\n--- 1. 经典 Gram-Schmidt 分解 ---")
    try:
        Q_gs, R_gs = classic_gram_schmidt(A)
        
        # 计算误差
        err_dec_gs = np.linalg.norm(A - Q_gs @ R_gs, 'fro')
        err_orth_gs = np.linalg.norm(Q_gs.T @ Q_gs - np.eye(Q_gs.shape[1]), 'fro')
        
        results.append({
            "Method": "Classic Gram-Schmidt",
            "Decomposition Error (err_dec)": err_dec_gs,
            "Orthogonality Error (err_orth)": err_orth_gs
        })
        
        print(f"分解有效性误差 ||A - QR||_fro: {err_dec_gs:.4e}")
        print(f"Q 矩阵正交性误差 ||Q^T*Q - I||_fro: {err_orth_gs:.4e}")
        if err_orth_gs > F_NORM_THRESHOLD:
            print(f"结论: 正交性误差 > {F_NORM_THRESHOLD}，Q矩阵不正交，算法不稳定。")
        else:
            print(f"结论: 正交性误差 <= {F_NORM_THRESHOLD}，Q矩阵正交，算法稳定。")

    except np.linalg.LinAlgError as e:
        print(f"QR分解失败: {e}")

    # --- 方法二: 修正的 Gram-Schmidt ---
    print("\n--- 2. 修正的 Gram-Schmidt 分解 ---")
    try:
        Q_mgs, R_mgs = modified_gram_schmidt(A)
        
        # 计算误差
        err_dec_mgs = np.linalg.norm(A - Q_mgs @ R_mgs, 'fro')
        err_orth_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(Q_mgs.shape[1]), 'fro')
        
        results.append({
            "Method": "Modified Gram-Schmidt",
            "Decomposition Error (err_dec)": err_dec_mgs,
            "Orthogonality Error (err_orth)": err_orth_mgs
        })
        
        print(f"分解有效性误差 ||A - QR||_fro: {err_dec_mgs:.4e}")
        print(f"Q 矩阵正交性误差 ||Q^T*Q - I||_fro: {err_orth_mgs:.4e}")
        if err_orth_mgs > F_NORM_THRESHOLD:
            print(f"结论: 正交性误差 > {F_NORM_THRESHOLD}，Q矩阵不正交，算法不稳定。")
        else:
            print(f"结论: 正交性误差 <= {F_NORM_THRESHOLD}，Q矩阵正交，算法稳定。")

    except np.linalg.LinAlgError as e:
        print(f"QR分解失败: {e}")

    # --- 方法三: Householder ---
    print("\n--- 3. Householder 分解 (Numpy) ---")
    try:
        Q_hh, R_hh = householder_qr(A)
        
        # 计算误差
        err_dec_hh = np.linalg.norm(A - Q_hh @ R_hh, 'fro')
        err_orth_hh = np.linalg.norm(Q_hh.T @ Q_hh - np.eye(Q_hh.shape[1]), 'fro')
        
        results.append({
            "Method": "Householder (Numpy)",
            "Decomposition Error (err_dec)": err_dec_hh,
            "Orthogonality Error (err_orth)": err_orth_hh
        })
        
        print(f"分解有效性误差 ||A - QR||_fro: {err_dec_hh:.4e}")
        print(f"Q 矩阵正交性误差 ||Q^T*Q - I||_fro: {err_orth_hh:.4e}")
        if err_orth_hh > F_NORM_THRESHOLD:
            print(f"结论: 正交性误差 > {F_NORM_THRESHOLD}，Q矩阵不正交，算法不稳定。")
        else:
            print(f"结论: 正交性误差 <= {F_NORM_THRESHOLD}，Q矩阵正交，算法稳定。")

    except np.linalg.LinAlgError as e:
        print(f"QR分解失败: {e}")

    # --- 结果汇总与保存 ---
    if results:
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df["Condition Number of A"] = cond_A
            
            # 重新排列列顺序
            df = df[["Method", "Condition Number of A", "Decomposition Error (err_dec)", "Orthogonality Error (err_orth)"]]
            
            output_csv_path = os.path.join(script_dir, 'qr_comparison_results.csv')
            df.to_csv(output_csv_path, index=False, float_format='%.4e')
            
            print(f"\n--- 结果已保存到: {output_csv_path} ---")
            print(df.to_string())
        except ImportError:
            print("\n警告: 未安装 pandas 库。结果无法保存到CSV文件。")
            print("请运行 'pip install pandas' 来安装。")
            print("\n--- 结果摘要 ---")
            for res in results:
                print(res)
    
if __name__ == '__main__':
    main()

