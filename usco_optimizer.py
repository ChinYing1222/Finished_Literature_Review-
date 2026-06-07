import numpy as np
from scipy.stats import qmc

class USCOOptimizer:
    def __init__(self, obj_func, lb, ub, max_evals=500, m=5, a_const=0.05, b_const=0.5, q_const=1.0, alpha=0.5):
        """
        USCO 優化器
        obj_func: 目標函數 (尋找此函數的最小值)
        lb/ub: 參數的下限與上限
        max_evals: 最大疊代次數 (T)
        m: 連續幾次未改善則觸發「跳脫機制」
        alpha: 探索與開發階段的分界比例
        """
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = len(lb)
        self.max_evals = max_evals
        self.m = m
        self.a_const = a_const
        self.b_const = b_const
        self.q_const = q_const
        self.alpha = alpha
        
        self.gbest = None         # 歷史最佳參數位置
        self.gbest_fit = float('inf') # 最佳適應度 (分數越低越好)
        self.c_count = 0          # 連續無改善計數器

    def optimize(self):
        # 1. 均勻初始化：用 Sobol 產生 0~1 的分佈 (Section 3.3 公式 15)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_sample = sampler.random(n=1)[0]
        
        # 映射：將 0~1 的比例轉換為實際參數範圍
        x = sobol_sample * (self.ub - self.lb) + self.lb
        self.gbest = np.copy(x)
        self.gbest_fit = self.obj_func(x) # 計算初始適應度
        
        history = [self.gbest_fit]
        t = 1
        split_eval = int(self.alpha * self.max_evals)
        
        while t < self.max_evals:
            # 2. 自適應權重：步長隨時間動態縮小 (Section 3.3 公式 16)
            std_dim = np.std(x) if self.dim > 1 else 0.001
            w = (self.q_const * np.exp(-self.b_const * std_dim)) / (1 + np.exp(self.a_const * (t - self.max_evals / 2)))
            
            x_new = np.copy(x)
            
            # 3. 逃脫機制：卡在死胡同時往遠處跳躍 (Section 3.2 公式 12)
            if self.c_count >= self.m:
                r3 = np.random.uniform(0, 1, self.dim)
                x_new = np.where(r3 < 0.5, 
                                 self.gbest + r3 * (self.ub - self.lb), 
                                 self.gbest - r3 * (self.ub - self.lb))
                self.c_count = 0
            else:
                # 4. 兩階段更新位置 (Section 3.2 公式 9 & 11)
                if t < split_eval:
                    # 第一階段：前期大步探索
                    r1 = np.random.uniform(0, 1, self.dim)
                    x_new = np.where(r1 < 0.5, 
                                     self.gbest + w * np.abs(self.gbest), 
                                     self.gbest - w * np.abs(self.gbest))
                else:
                    # 第二階段：後期小步定點開發
                    r2 = np.random.uniform(0, 1, self.dim)
                    x_new = np.where(r2 < 0.5, 
                                     self.gbest + r2 * w * (self.ub - self.lb), 
                                     self.gbest - r2 * w * (self.ub - self.lb))
            
            # 5. 邊界約束：越界則直接拉回已知最優位置 (Section 3.2 公式 13)
            out_of_bounds = (x_new < self.lb) | (x_new > self.ub)
            x_new[out_of_bounds] = self.gbest[out_of_bounds]
            
            # 6. 計算新適應度並更新最佳解
            fit_new = self.obj_func(x_new)
            
            if fit_new < self.gbest_fit:
                self.gbest_fit = fit_new
                self.gbest = np.copy(x_new)
                x = np.copy(x_new)
                self.c_count = 0
            else:
                self.c_count += 1
                x = np.copy(x_new) # 接受更新以維持搜尋活力
                
            history.append(self.gbest_fit)
            t += 1
            
        return self.gbest, self.gbest_fit, history