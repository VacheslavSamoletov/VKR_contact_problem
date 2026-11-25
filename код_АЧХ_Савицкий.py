# -*- coding: utf-8 -*-
"""
амплитудно-частотная характеристика δ(κ) штампа на ФГМ-слое.

Для каждого κ из заданного диапазона:
1. Создаётся новый FGMLayer и Integral (\kappa зашит в обоих)
2. Собирается матрица МГЭ и решается СЛАУ для τ_0
3. По формуле δ = T / (∫τ_0 - m·κ²) считается смещение
4. Результат накапливается в массив δ(κ)
Строится график Re δ, Im δ 
Время: при N = 20 одно κ ≈ 5-10 секунд, для N_kappa сетка из 20-30 точек ≈ 7-10 минут.
для N = 40 и сетке N_kappa = 80 точность высокая, но и расчет долгий (примерно минут 30)
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from kernel_fgm_A import FGMLayer, Integral


# =====================================================================
#  сборка и решение СЛАУ метода граничных элементов для нормированной задачи ∫−aak(x1−ξ) τ0(ξ) dξ=1\int_{-a}^a k(x_1-\xi)\,\tau_0(\xi)\,d\xi=1
# ∫{−a}{a}k(x1−ξ)τ0(ξ)dξ=1 (единичное смещение штампа)
# (J_vals = [integral.evaluate(u) ...] -- значения первообразной J(u) в этих точках. Каждое evaluate это 
# контурный интеграл с задачами Коши RK4 внутри. 
# Сборка первого столбца Тёплицевой матрицы c[0] = 2·J_vals[0] c[k] = J_vals[k]-J_vals[k-1]
#A = toeplitz(c, c) — матрица симметрично-Тёплицева
#solve(A, ones) - τ0
#τ0 - распределение контактных напряжений на единичное смещение.
# =====================================================================

def solve_tau0(integral, N, a_half):
    """Сборка матрицы + решение СЛАУ для нормированной задачи (RHS = 1)."""
    h = 2.0 * a_half / N
    ys = h * (np.arange(N) + 0.5)
    J_vals = np.array([integral.evaluate(float(u)) for u in ys])

    c = np.empty(N, dtype=complex)
    c[0] = 2.0 * J_vals[0]
    c[1:] = J_vals[1:] - J_vals[:-1]
    A = toeplitz(c, c)

    rhs = np.ones(N, dtype=complex)
    tau_0 = np.linalg.solve(A, rhs)
    return tau_0, h


# =====================================================================
# Одна точка АЧХ
# layer = FGMLayer(... epsilon=1e-3, n_steps=200) — слой на частоте \kappa
# с предельным поглощением
#ε=10−3 (сдвигает полюсы/точку ветвления с оси) и 200 шагами RK4 для задач Коши.
#alpha_end = 1.2·κ/√μ_min — верхний предел интегрирования по \alpha
# Полюсы (корни дисперсии) лежат в (\kappa,\ \kappa/\sqrt{\mu_{\min}})
# множитель 1.2 даёт 20% запаса за последним полюсом. То есть граница интеграла задаётся именно дисперсионной полосой
#afe_integrand — обёртка над K_over_alpha с защитой от inf/nan (возле полюсов и при больших \alpha
# где cosh/sinh переполняются) — возвращает 0j вместо краха
# integral = Integral(...) → solve_tau0 → τ0
# integral_tau_0 = 1j·h·Σ τ_0 — интеграл ∫τ0 dξ по формуле прямоугольников,
# ×i —  нормировка ядра.
# delta = T/(integral_tau_0 - m κ²) — уравнение движения штампа.
# =====================================================================

def compute_delta_at_kappa(kappa, mu_func, rho_func, a_half, N, T, mass,
                            epsilon=1e-3, n_steps=200, n_sub=10):
    """Возвращает δ и ∫τ_0 dξ для заданного κ."""
    layer = FGMLayer(mu_func=mu_func, rho_func=rho_func,
                     kappa=kappa, epsilon=epsilon, n_steps=n_steps)
    alpha_end = 1.2 * kappa / math.sqrt(layer.mu_min)

    def safe_integrand(alpha):
        try:
            K = layer.K_over_alpha(alpha)
            if not (math.isfinite(K.real) and math.isfinite(K.imag)):
                return 0j
            return K
        except (OverflowError, ValueError):
            return 0j

    integral = Integral(safe_integrand, kappa=kappa,
                        alpha_end=alpha_end, n_sub=n_sub)

    tau_0, h = solve_tau0(integral, N, a_half)

#     integral_tau_0 = h * np.sum(tau_0)
#     delta = T / (-integral_tau_0 - mass * kappa * kappa)
    integral_tau_0 = 1j * h * np.sum(tau_0)      # нормировка ядра: ∫τ0_физ = i·∫τ0_код
    delta = T / (integral_tau_0 - mass * kappa ** 2)
    return delta, integral_tau_0

# =====================================================================
# Main: сканирование по \kappa и построение АЧХ
# =====================================================================

def main():
    # ---------- Параметры ----------
    mu_func  = lambda x: 1.0 - 0.8 * x         # мягкий верх
    rho_func = lambda x: 1.0
    a_half = 0.1
    T = 1.0 + 0j
    mass = 0.05
    N = 30                                      

    # ---------- Диапазон частот ----------
    kappa_min = 1.5
    kappa_max = 10.0
    n_kappa = 80
    kappas = np.linspace(kappa_min, kappa_max, n_kappa)

    print(f"АЧХ: ФГМ-слой «мягкий верх», μ(x) = 1 - 0.8x, ρ = 1")
    print(f"  T = {T},  m = {mass},  N = {N},  a = {a_half}")
    print(f"  κ ∈ [{kappa_min}, {kappa_max}],  точек: {n_kappa}")
    print(f"  Оценочное время ~{n_kappa * 6} секунд\n")

    # ---------- Расчёт ----------
    deltas = np.zeros(n_kappa, dtype=complex)
    integrals = np.zeros(n_kappa, dtype=complex)

    t0 = time.time()
    for i, kappa in enumerate(kappas):
        delta_i, integral_i = compute_delta_at_kappa(
            kappa, mu_func, rho_func, a_half, N, T, mass)
        deltas[i] = delta_i
        integrals[i] = integral_i

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_kappa - i - 1) / (i + 1)
            print(f"  {i+1}/{n_kappa}  κ={kappa:.3f}  "
                  f"δ={delta_i.real:+.3e}{delta_i.imag:+.3e}j  "
                  f"({elapsed:.0f}s, ост: {eta:.0f}s)")

    total = time.time() - t0
    print(f"\nГотово за {total:.1f} с ({total / n_kappa:.1f} с/точку)")


    # ---------- Сохраняем числовые данные ----------
    #  Текстовый 
    output_txt = 'afc_data777.txt'
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("# Результаты расчета АЧХ штампа на ФГМ-слое\n")
        f.write(f"# Параметры: N={N}, a={a_half}, m={mass}, T={T.real}\n")
        f.write("kappa\tdelta_Re\tdelta_Im\t|delta| (Amp)\n")
        
        for k, d in zip(kappas, deltas):
            amp = np.abs(d)
            f.write(f"{k:.4f}\t{d.real:.8e}\t{d.imag:.8e}\t{amp:.8e}\n")
            
    print(f"✓ Текстовая таблица успешно сохранена в {output_txt}")

    # ---------- Графики ----------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Верхний - Re и Im δ
    axes[0].plot(kappas, deltas.real, 'k-',  lw=1.4, label=r'Re $\delta$')
    axes[0].plot(kappas, deltas.imag, 'k--', lw=1.1, label=r'Im $\delta$')
    axes[0].axhline(0, color='gray', lw=0.4, alpha=0.5)
    axes[0].set_ylabel(r'$\delta$', fontsize=12)
    axes[0].set_title(
        rf'АЧХ штампа на ФГМ-слое: $\delta(\kappa)$  '
        rf'$\mu(x_2) = 1 - 0.8 x_2$, '
        rf'$T = {T.real}$, $m = {mass}$, $N = {N}$',
        fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')

    # Нижний - модуль на лог-шкале 
    axes[1].semilogy(kappas, np.abs(deltas), 'k-', lw=1.4)
    axes[1].set_xlabel(r'$\kappa$', fontsize=12)
    axes[1].set_ylabel(r'$|\delta|$', fontsize=12)
    axes[1].set_title('Модуль |δ(κ)| (логарифмическая шкала)', fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')

    plt.tight_layout()
#     plt.savefig('afc_delta.png', dpi=200, bbox_inches='tight')
#     print("✓ График сохранён в afc_delta.png")
    plt.show()
    

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(kappas, integrals.real, 'k-', lw=1.5, label='Re ∫τ₀')
    ax.plot(kappas, integrals.imag, 'k--', lw=1.2, label='Im ∫τ₀')
    ax.set_xlabel('κ'); ax.set_ylabel('∫τ₀(ξ)dξ')
    ax.set_title('Интегральная характеристика системы (без T, m)')
    ax.grid(True, alpha=0.3); ax.legend()
#     plt.savefig('integral_tau0.png', dpi=200)

if __name__ == "__main__":
    main()
