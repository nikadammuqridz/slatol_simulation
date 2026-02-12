"""
SLATOL Main Experiment
FIXED: Added interactive debug mode, figure handling
Nik Adam Muqridz (2125501)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from simulation import run_single_trial

MASS_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
WIND_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TOTAL_MASS = 1.0
GRAVITY = 9.81

def run_debug():
    """Interactive debug: single trial with GUI kept open."""
    print("\n🔧 DEBUG MODE")
    try:
        mu = float(input("Enter leg mass ratio μ (0.05-0.30): "))
        eta = float(input("Enter wind ratio η (0.0-0.5): "))
    except:
        print("Invalid input. Using μ=0.20, η=0.20")
        mu, eta = 0.20, 0.20

    print(f"\nRunning single trial: μ={mu:.2f}, η={eta:.2f} with GUI...")
    res = run_single_trial(mu, eta, trial_id=999,
                           use_gui=True, debug=True, max_time=5.0)
    print(f"Result: {'SUCCESS' if res['success'] else 'FAIL'} ({res['failure_cause']})")
    print(f"Max pitch: {res['max_pitch']:.2f}°, Max height: {res['max_height']:.2f}m")
    if res['wind_applied']:
        print(f"Wind applied at t={res['wind_time']:.3f}s")

def run_sweep():
    """Full parameter sweep."""
    print("\n📊 Running full parameter sweep (36 trials)...")
    results = []
    trial_counter = 0

    for mu in MASS_RATIOS:
        for eta in WIND_RATIOS:
            trial_counter += 1
            print(f"Trial {trial_counter:2d}: μ={mu:.2f}, η={eta:.2f} ...", end='', flush=True)

            # Use GUI for first 3 trials only
            use_gui = (trial_counter <= 3)
            res = run_single_trial(mu, eta, trial_counter,
                                   use_gui=use_gui, debug=False, max_time=3.0)

            results.append(res)
            status = "✓" if res['success'] else "✗"
            print(f" {status}  max_pitch={res['max_pitch']:.1f}°, max_height={res['max_height']:.2f}m")

    # Save CSV
    csv_file = "pybullet_results_final.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mu', 'eta', 'success',
                                               'max_height', 'max_pitch',
                                               'settling_time', 'max_overshoot'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'mu': r['mu'],
                'eta': r['eta'],
                'success': r['success'],
                'max_height': r['max_height'],
                'max_pitch': r['max_pitch'],
                'settling_time': r['settling_time'],
                'max_overshoot': r['max_overshoot']
            })
    print(f"\nResults saved to {csv_file}")

    # Generate figures
    generate_figures(results)

def generate_figures(results):
    """Generate thesis figures from sweep results."""
    print("Generating figures...")

    # Figure 4.1: Nominal flight (μ=0.05, η=0.0)
    nom = next((r for r in results if r['mu']==0.05 and r['eta']==0.0), None)
    if nom and nom['success']:
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(nom['time'], nom['height'], 'b-', linewidth=2)
        plt.ylabel('CoM Height (m)')
        plt.title('Figure 4.1: Nominal Flight Performance (μ=0.05, η=0.00)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0,1.0)
        plt.subplot(2,1,2)
        plt.plot(nom['time'], nom['pitch'], 'r-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Angle (deg)')
        plt.grid(True, alpha=0.3)
        plt.ylim(-10,10)
        plt.tight_layout()
        plt.savefig('figure_4_1_nominal.png', dpi=300)
        plt.close()
        print("  ✅ Figure 4.1 saved")
    else:
        print("  ⚠️  No successful nominal trial – skipping Figure 4.1")

    # Figure 4.2: Stable recovery (μ=0.20, η=0.20)
    stable = next((r for r in results if r['mu']==0.20 and r['eta']==0.20), None)
    if stable and stable['success'] and stable['wind_applied']:
        plt.figure(figsize=(10,5))
        plt.plot(stable['time'], stable['pitch'], 'b-', linewidth=2, label='Pitch Response')
        # wind injection line
        wind_time = stable['wind_time']
        plt.axvline(x=wind_time, color='k', linestyle='--', alpha=0.7, label='Wind Injection')
        plt.axhline(y=2, color='g', linestyle=':', label='±2° Band')
        plt.axhline(y=-2, color='g', linestyle=':')
        if stable['settling_time'] is not None:
            settle_time = wind_time + stable['settling_time']
            plt.axvline(x=settle_time, color='r', linestyle='--', alpha=0.7,
                       label=f"Settling: {stable['settling_time']:.2f}s")
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Angle (deg)')
        plt.title('Figure 4.2: Stable Recovery (μ=0.20, η=0.20)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-30,30)
        plt.savefig('figure_4_2_stable.png', dpi=300)
        plt.close()
        print("  ✅ Figure 4.2 saved")
    else:
        print("  ⚠️  No successful stable trial – skipping Figure 4.2")

    # Figure 4.3: Unstable divergence
    # Try μ=0.20, η=0.4 or μ=0.25, η=0.0
    unstable = next((r for r in results if r['mu']==0.20 and r['eta']>0.3 and not r['success']), None)
    if not unstable:
        unstable = next((r for r in results if r['mu']==0.25 and r['eta']==0.0 and not r['success']), None)
    if unstable:
        plt.figure(figsize=(10,5))
        plt.plot(unstable['time'], unstable['pitch'], 'r-', linewidth=2, label='Pitch Response')
        plt.axhline(y=45, color='k', linestyle='--', linewidth=2, label='Failure Threshold (45°)')
        exceed_idx = np.where(np.abs(unstable['pitch']) > 45)[0]
        if len(exceed_idx) > 0:
            fail_time = unstable['time'][exceed_idx[0]]
            plt.axvline(x=fail_time, color='r', linestyle='--', alpha=0.7,
                       label=f'Failure at t={fail_time:.2f}s')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Angle (deg)')
        plt.title(f"Figure 4.3: Unstable Divergence (μ={unstable['mu']:.2f}, η={unstable['eta']:.2f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-10, 90)
        plt.savefig('figure_4_3_unstable.png', dpi=300)
        plt.close()
        print("  ✅ Figure 4.3 saved")
    else:
        print("  ⚠️  No failure trial found – skipping Figure 4.3")

    # Figure 4.4: Stability Sufficiency Map
    stability_limits = {}
    for mu in MASS_RATIOS:
        mu_results = [r for r in results if r['mu'] == mu and r['success']]
        if mu_results:
            max_eta = max(r['eta'] for r in mu_results)
            stability_limits[mu] = max_eta
        else:
            stability_limits[mu] = 0.0

    plt.figure(figsize=(12,8))
    for r in results:
        color = 'green' if r['success'] else 'red'
        plt.scatter(r['mu'], r['eta'], c=color, alpha=0.6, edgecolors='k', s=50)
    mu_vals = sorted(stability_limits.keys())
    eta_vals = [stability_limits[m] for m in mu_vals]
    plt.plot(mu_vals, eta_vals, 'k-', linewidth=3, label='Stability Boundary')
    plt.fill_between(mu_vals, 0, eta_vals, alpha=0.2, color='green',
                     label='Stability Sufficiency Region')
    plt.xlabel('Leg Mass Ratio (μ)', fontsize=14)
    plt.ylabel('Wind Disturbance Ratio (η)', fontsize=14)
    plt.title('Figure 4.4: Stability Sufficiency Region from PyBullet Simulation', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper right')
    plt.xlim(0, 0.35)
    plt.ylim(0, 0.6)
    plt.savefig('figure_4_4_stability_map.png', dpi=300)
    plt.close()
    print("  ✅ Figure 4.4 saved")

    # Print Table 4.1
    print("\n" + "="*60)
    print("TABLE 4.1: Stability Limits and Quality Metrics")
    print("="*60)
    print(f"{'μ':<8} {'η_max':<10} {'Settling Time (s)':<20} {'Max Overshoot (°)':<20}")
    print("-"*60)
    for mu in MASS_RATIOS:
        eta_max = stability_limits[mu]
        rep = next((r for r in results if r['mu']==mu and r['eta']==eta_max and r['success']), None)
        if rep:
            settle = f"{rep['settling_time']:.3f}" if rep['settling_time'] else 'N/A'
            overshoot = f"{rep['max_overshoot']:.1f}" if rep['max_overshoot'] else 'N/A'
            print(f"{mu:.2f}    {eta_max:.3f}       {settle:<20} {overshoot:<20}")
        else:
            print(f"{mu:.2f}    {eta_max:.3f}       {'N/A':<20} {'N/A':<20}")
    print("="*60)

if __name__ == "__main__":
    print("SLATOL Experiment - FIXED VERSION")
    print("="*60)
    print("1. Run full parameter sweep")
    print("2. Debug mode (single trial with GUI)")
    choice = input("Select mode (1/2): ").strip()

    if choice == '2':
        run_debug()
    else:
        run_sweep()