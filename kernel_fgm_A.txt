# -*- coding: utf-8 -*-
"""
kernel_fgm_A.py — вариант A: численное вычисление J(u) для ФГМ-слоя без
каких-либо асимптотических приближений в хвосте.

Единственные две правки:
  1. Аналитическая equation_symbol(α) заменена на FGMLayer.K_over_alpha(α),
     которая решает задачу Коши методом RK4 в комплексной арифметике.
  2. Контурный участок расширен с [0, 2κ] до [0, α_end], где
     α_end = 1.2 · κ / √μ_min — чтобы охватить полюса для мягкого ФГМ
   

В хвосте по вещественной оси используется рассчитанное RK4 значение K̃ ,
а не асимптотика.

"""

import cmath
import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


PI = 3.1415926538
I = 1j

XI = (
    -0.989400934991650, -0.944575023073233, -0.865631202387832,
    -0.755404408355003, -0.617876244402644, -0.458016777657227,
    -0.281603550779259, -0.0950125098376374,
     0.0950125098376374,  0.281603550779259,  0.458016777657227,
     0.617876244402644,  0.755404408355003,  0.865631202387832,
     0.944575023073233,  0.989400934991650
)

ETA = (
    0.0271524594117540, 0.0622535239386475, 0.0951585116824939,
    0.124628971255534,  0.149595988816577,  0.169156519395002,
    0.182603415044922,  0.189450610455068,
    0.189450610455068,  0.182603415044922,  0.169156519395002,
    0.149595988816577,  0.124628971255534,  0.0951585116824939,
    0.0622535239386475, 0.0271524594117540
)


# =====================================================================
# Класс FGMLayer — замена equation_symbol для ФГМ
# =====================================================================

class FGMLayer:

    def __init__(self,
                 mu_func: Callable[[float], float],
                 rho_func: Callable[[float], float],
                 kappa: float,
                 epsilon: float = 1e-3,
                 n_steps: int = 200):
        self.mu = mu_func
        self.rho = rho_func
        self.kappa_real = float(kappa)

        # Предельное поглощение κ → κ(1+iε)
        self.kappa = complex(kappa) * (1 + 1j * epsilon)

        self.n_steps = n_steps

        self.mu_min = min(mu_func(k / 50) for k in range(51))

    def _rhs(self, x, U, S, alpha):
        m = self.mu(x)
        r = self.rho(x)
        dU = S / m
        dS = (alpha * alpha * m - r * self.kappa * self.kappa) * U
        return dU, dS

    def _cauchy_one(self, alpha, U0, S0):

        """RK4 от x_2 = 0 до x_2 = 1 с одной парой начальных условий."""

        h = 1.0 / self.n_steps
        U, S = U0, S0
        x = 0.0
        for _ in range(self.n_steps):
            k1U, k1S = self._rhs(x, U, S, alpha)
            k2U, k2S = self._rhs(x + h/2, U + 0.5*h*k1U, S + 0.5*h*k1S, alpha)
            k3U, k3S = self._rhs(x + h/2, U + 0.5*h*k2U, S + 0.5*h*k2S, alpha)
            k4U, k4S = self._rhs(x + h,   U + h*k3U,    S + h*k3S,    alpha)
            U = U + (h / 6.0) * (k1U + 2*k2U + 2*k3U + k4U)
            S = S + (h / 6.0) * (k1S + 2*k2S + 2*k3S + k4S)
            x = x + h
        return U, S

    def K_over_alpha(self, alpha: complex) -> complex:
        gamma2 = cmath.sqrt(alpha * alpha - self.kappa * self.kappa)
        U1H, S1H = self._cauchy_one(alpha, 1.0 + 0j, 0.0 + 0j)
        U2H, S2H = self._cauchy_one(alpha, 0.0 + 0j, 1.0 + 0j)
        K = (U1H + gamma2 * U2H) / (S1H + gamma2 * S2H)
        return K / alpha




class Integral:
    """
    Вычисление J(u) = (i/π) ∫_σ K̃(α) sin(αu) dα/α  через:
      1. контур Зоммерфельда на [0, α_end] (дуга вниз на глубину δ)
      2. вещественная ось на [α_end, ∞), адаптивно по периодам sin(αu)
    """

    def __init__(self, integrand, kappa, alpha_end, n_sub=10):
        self._integrand = integrand
        self._kappa = kappa
        self._alpha_end = alpha_end
        self._delta = kappa / 5.0          
        self._n_sub = n_sub                 
        self._mid = alpha_end / 2.0

    def _gauss_simple(self, a, b, x):
        m = 0.5 * (a + b)
        n = 0.5 * (b - a)
        result = 0j
        for i in range(16):
            y = m + n * XI[i]
            result += ETA[i] * self._integrand(y) * cmath.sin(y * x)
        return result

    def _gauss_contour(self, a, b, x, contour, derivative):
    #Гаусс-Лежандр с параметризованным комплексным контуром α(y).
        m = 0.5 * (a + b)
        n = 0.5 * (b - a)
        result = 0j
        for i in range(16):
            y = m + n * XI[i]
            alpha = contour(y)
            da = derivative(y)
            result += ETA[i] * self._integrand(alpha) * cmath.sin(alpha * x) * da
        return result

    def evaluate(self, x):
        
        # Возвращает J(x)
      
        result = 0j
        delta = self._delta
        alpha_end = self._alpha_end
        mid = self._mid

        # Часть 1: деформированный контур [0, α_end]
        # Парабола α(y) = y - i·δ·(1 - ((y-mid)/mid)²),  y ∈ [0, α_end].
        # Касается ℝ в y=0 и y=α_end, опускается на δ при y=mid.

        def contour(y):
            r = (y - mid) / mid
            return complex(y, -delta * (1.0 - r * r))

        def derivative(y):
            return complex(1.0, 2.0 * delta * (y - mid) / (mid * mid))

        sub_width = alpha_end / self._n_sub
        for j in range(self._n_sub):
            a = sub_width * j
            b = sub_width * (j + 1)
            result += self._gauss_contour(a, b, x, contour, derivative)
        result *= 0.5 * sub_width  

        # Часть 2: вещественная ось [α_end, ∞), адаптивно
        
        if abs(x) > 1e-12:
            segment = abs(2.0 * PI / x)
            x0 = alpha_end
            for _ in range(1000):     # защита от бесконечного цикла
                term = self._gauss_simple(x0, x0 + segment, x)
                result += term * 0.5 * segment
                x0 += segment
                if abs(term) <= 1e-5:
                    break

        return I * result / PI




def main():
    # Параметры задачи
    kappa = 3.0
    eps = 1e-3
    mu_func = lambda x: 1.0 - 0.8 * x   # мягкий верх
    rho_func = lambda x: 1.0

    # ФГМ-слой
    layer = FGMLayer(mu_func=mu_func, rho_func=rho_func,
                     kappa=kappa, epsilon=eps, n_steps=200)
    print(f"ФГМ-слой:")
    print(f"  μ(0) = {mu_func(0):.2f},  μ(1) = {mu_func(1):.2f}")
    print(f"  μ_min = {layer.mu_min:.3f}")
    print(f"  κ = {kappa},  ε = {eps}")

    alpha_end = 1.2 * kappa / math.sqrt(layer.mu_min)
    print(f"  α_end = {alpha_end:.3f}  (= 1.2·κ/√μ_min)")

   
    integrand = lambda alpha: layer.K_over_alpha(alpha)

    integral = Integral(integrand, kappa=kappa,
                        alpha_end=alpha_end, n_sub=10)

    
    u_min, u_max, n_points = -2.0, 2.0, 50
    u_values = np.linspace(u_min, u_max, n_points)
    results = np.zeros(n_points, dtype=complex)

    print(f"\nВычисление J(u) на {n_points} точках "
          f"(каждая = ~1000 RK4-задач)...")
    import time
    t0 = time.time()
    for idx, u in enumerate(u_values):
        results[idx] = integral.evaluate(float(u))
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{n_points}  ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\nГотово за {total_time:.1f}s.")
    print(f"Статистика:")
    print(f"  Re: min={results.real.min():+.4e},  max={results.real.max():+.4e}")
    print(f"  Im: min={results.imag.min():+.4e},  max={results.imag.max():+.4e}")
    print(f"  |J|: max={np.abs(results).max():.4e}")

    # ============= Визуализация =============
    plt.figure(figsize=(12, 7))

    plt.plot(u_values, results.real,
             label='Re[J(u)]',
             linewidth=2.5, color='#2E86AB',
             marker='o', markersize=4, markevery=5)
    plt.plot(u_values, results.imag,
             label='Im[J(u)]',
             linewidth=2.5, color='#A23B72',
             linestyle='--',
             marker='s', markersize=4, markevery=5)

    plt.xlabel('Параметр u', fontsize=13, fontweight='bold')
    plt.ylabel('Значение интеграла J(u)', fontsize=13, fontweight='bold')
    plt.title(r'Вариант A: J(u) для ФГМ-слоя (RK4 везде, без асимптотик)'
              f'\n$\\mu(x_2) = 1 - 0.8 x_2$, $\\kappa = {kappa}$, '
              f'$\\varepsilon = {eps}$',
              fontsize=13, fontweight='bold', pad=15)

    plt.grid(True, linestyle=':', alpha=0.6, linewidth=0.8)
    plt.legend(fontsize=11, frameon=True, shadow=True)
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
    plt.axvline(0, color='gray', linestyle='-', linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    plt.savefig('J_fgm_variantA.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ График сохранён в J_fgm_variantA.png")
    plt.show()


if __name__ == "__main__":
    main()
