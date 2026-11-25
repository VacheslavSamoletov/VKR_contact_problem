#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
"""
dispersion.py — дисперсионные кривые волн Лява для ФГМ-слоя.

Дисперсионное уравнение получается из требования существования
нетривиального решения вспомогательной задачи без приложенной нагрузки:
    D(α, κ) = Σ_1(1; α, κ) + γ_2(α, κ) Σ_2(1; α, κ) = 0,
где γ_2 = √(α² - κ²) (вещественная положительная для α > κ),
а Σ_j — фундаментальные решения системы ОДУ из 2statement.

Поверхностные моды существуют в полосе  κ < α < κ/√μ_min,
что соответствует фазовой скорости c = κ/α в интервале  √μ_min < c < 1.
Для μ(x) = 1 - 0.8x:  μ_min = 0.2, c ∈ (0.45, 1.0).

Для каждого κ находим все нули D(α, κ) методом отслеживания знаков
+ Brent. Затем собираем точки в непрерывные ветви, отслеживая их 
траектории с помощью функции assemble_branches.
"""

import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from kernel_fgm_A import FGMLayer

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def save_dispersion_points(branches, filename="dispersion_points2.txt"):
    """
    Сохраняет вычисленные координаты точек дисперсионных кривых в текстовый файл.
    Формат: Номер_моды   Частота_(kappa)   Волновое_число_(alpha)   Скорость_(c)
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Точки дисперсионных кривых волн Лява в ФГМ-слое\n")
        f.write("# Структура: Разделение табуляцией (TSV)\n")
        f.write("Mode\tKappa\tAlpha\tPhaseVelocity_c\n")
        
        for n, branch in enumerate(branches):
            f.write(f"# --- Данные для Моды № {n + 1} ---\n")
            for kappa, alpha in branch:
                c_velocity = kappa / alpha
                f.write(f"{n + 1}\t{kappa:.6f}\t{alpha:.6f}\t{c_velocity:.6f}\n")
            f.write("\n")
            
    print(f"✓ Координаты точек дисперсионных кривых сохранены в файл: {filename}")

def compute_D_real(alpha, kappa, mu_func, rho_func, n_steps=200):
    """
    Дисперсионная функция D(α, κ) для вещественных α > κ > 0.
    Возвращает вещественное число (мнимая часть в пределе округления).
    """
    if alpha <= kappa:
        return float('nan')

    layer = FGMLayer(mu_func=mu_func, rho_func=rho_func,
                     kappa=kappa, epsilon=0.0, n_steps=n_steps)
    gamma_2 = math.sqrt(alpha * alpha - kappa * kappa)
    _, S1 = layer._cauchy_one(complex(alpha), 1.0 + 0j, 0.0 + 0j)
    _, S2 = layer._cauchy_one(complex(alpha), 0.0 + 0j, 1.0 + 0j)
    D = S1 + gamma_2 * S2
    return D.real

def find_modes_at_kappa(kappa, mu_func, rho_func, mu_min,
                         n_grid=200, n_steps=200):
    """
    Возвращает отсортированный список α_n с D(α_n, κ) = 0
    на интервале (κ, κ/√μ_min · 1.02).
    """
    alpha_min = kappa * 1.0005
    alpha_max = kappa / math.sqrt(mu_min) * 1.02

    alphas = np.linspace(alpha_min, alpha_max, n_grid)
    D_values = np.array([
        compute_D_real(a, kappa, mu_func, rho_func, n_steps) for a in alphas
    ])

    modes = []
    for i in range(len(alphas) - 1):
        d1, d2 = D_values[i], D_values[i + 1]
        if not (math.isfinite(d1) and math.isfinite(d2)):
            continue
        if d1 * d2 < 0:
            try:
                root = brentq(
                    lambda a: compute_D_real(a, kappa, mu_func, rho_func, n_steps),
                    alphas[i], alphas[i + 1], xtol=1e-6, maxiter=50)
                modes.append(root)
            except (ValueError, RuntimeError):
                pass
    return sorted(modes)

def assemble_branches(kappas, all_modes, tol=0.6):
    """
    Связывает корни в непрерывные физические ветви.
    Возвращает список ветвей, каждая — список (kappa, alpha),
    отсортированный по частоте отсечки.
    """
    branches = []
    for kappa, modes in zip(kappas, all_modes):
        used = [False] * len(modes)
        for br in branches:
            if len(br) >= 2:
                (k0, a0), (k1, a1) = br[-2], br[-1]
            else:
                (k0, a0), (k1, a1) = br[-1], br[-1]
                
            slope = (a1 - a0) / (k1 - k0) if len(br) >= 2 and k1 != k0 else 1.0
            pred = br[-1][1] + slope * (kappa - br[-1][0])
            best, bestd = -1, 1e9
            for j, a in enumerate(modes):
                if not used[j] and abs(a - pred) < bestd:
                    bestd, best = abs(a - pred), j
            if best >= 0 and bestd < tol:
                br.append((kappa, modes[best]))
                used[best] = True
        for j, a in enumerate(modes):
            if not used[j]:
                branches.append([(kappa, a)])
    branches.sort(key=lambda b: b[0][0])
    return branches

def main():
    sys.stdout = DualLogger("dispersion_log2.txt")
    
    mu_func = lambda x: 1.0 - 0.8 * x
    rho_func = lambda x: 1.0
    mu_min = 0.2

    print("=" * 65)
    print("Дисперсионные кривые волн Лява в ФГМ-слое")
    print("=" * 65)
    print(f"  μ(x) = 1 - 0.8x,  ρ(x) = 1")
    print(f"  μ_min = {mu_min},  c_layer ∈ ({math.sqrt(mu_min):.3f}, 1.000)")
    print()

    kappa_min, kappa_max, n_kappa = 0.5, 50.0, 100
    kappas = np.linspace(kappa_min, kappa_max, n_kappa)

    all_modes = []
    t0 = time.time()

    for i, kappa in enumerate(kappas):
        modes = find_modes_at_kappa(kappa, mu_func, rho_func, mu_min,
                                     n_grid=200, n_steps=200)
        all_modes.append(modes)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  {i + 1:>3}/{n_kappa}  κ = {kappa:5.2f}  "
                  f"мод: {len(modes)}  ({time.time() - t0:.0f}s)")

    total_time = time.time() - t0
    
    # Сборка непрерывных ветвей
    branches = assemble_branches(kappas, all_modes)
    print(f"\nГотово за {total_time:.1f} с.  Всего непрерывных ветвей: {len(branches)}")

    # Сохранение точек
    save_dispersion_points(branches, "dispersion_points2.txt")

    # Графики
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Слева: (κ, α)
    kk = np.array([0, kappa_max])
    axes[0].plot(kk, kk, 'k:', lw=1, alpha=0.5, label=r'$\alpha = \kappa$ (граница излучения)')
    axes[0].plot(kk, kk / math.sqrt(mu_min), 'k--', lw=1, alpha=0.5, label=r'$\alpha = \kappa/\sqrt{\mu_{\min}}$')

    for n, branch in enumerate(branches):
        ks = [pt[0] for pt in branch]
        als = [pt[1] for pt in branch]
        axes[0].plot(ks, als, '-', lw=1.2, color='maroon', label=f'Mode {n + 1}' if n < 5 else "")
        
    axes[0].set_xlabel(r'$\kappa$', fontsize=12)
    axes[0].set_ylabel(r'$\alpha$', fontsize=12)
    axes[0].set_title(r'Дисперсионные кривые в осях $(\kappa, \alpha)$', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    # Ограничиваем легенду только первыми 5 модами, чтобы не перегружать график
    axes[0].legend(fontsize=8, loc='upper left') 
    axes[0].set_xlim(0, kappa_max)

    # Справа: (α, c)
    axes[1].axhline(math.sqrt(mu_min), color='gray', ls=':', lw=1, alpha=0.6,
                    label=rf'$c = \sqrt{{\mu_{{\min}}}} = {math.sqrt(mu_min):.2f}$')
    axes[1].axhline(1.0, color='gray', ls='--', lw=1, alpha=0.6,
                    label=r'$c = c_{\mathrm{substrate}} = 1$')

    for n, branch in enumerate(branches):
        als = [pt[1] for pt in branch]
        cs = [pt[0] / pt[1] for pt in branch]
        axes[1].plot(als, cs, '-', lw=1.2, color='maroon', label=f'Mode {n + 1}' if n < 5 else "")

    axes[1].set_xlabel(r'$\alpha$', fontsize=12)
    axes[1].set_ylabel(r'$c = \kappa / \alpha$', fontsize=12)
    axes[1].set_title(r'Фазовая скорость в осях $(\alpha, c)$', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc='best')
    axes[1].set_ylim(0.4, 1.05)

    fig.suptitle(r'Дисперсионные кривые волн Лява для ФГМ-слоя $\mu(x_2) = 1 - 0.8 x_2$', fontsize=13)
    plt.tight_layout()
    plt.savefig('dispersion_curves2.png', dpi=200, bbox_inches='tight')
    print("✓ График сохранён в dispersion_curves2.png")
    plt.show()

if __name__ == "__main__":
    main()
