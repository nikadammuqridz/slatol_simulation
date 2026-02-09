#!/usr/bin/env python3
"""
SLATOL Simulation Entry Point
Run with: python run_simulation.py [mode]
"""

import sys
import argparse
from slatol_simulator import SLATOLSimulator
from slatol_sweep import StabilitySweep

def main():
    parser = argparse.ArgumentParser(description='SLATOL Simulation')
    parser.add_argument('mode', choices=['gui', 'sweep', 'test'],
                       help='Simulation mode')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Simulation duration (seconds)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for sweep')
    parser.add_argument('--save', type=str,
                       help='Save results to file')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        # Interactive GUI mode
        print("Starting interactive GUI simulation...")
        sim = SLATOLSimulator(gui=True)
        sim.initialize_robot()
        sim.create_gui()
    
    elif args.mode == 'sweep':
        # Automated parameter sweep
        print("Starting stability region sweep...")
        sweep = StabilitySweep()
        
        # Run sweep
        results = sweep.run_sweep(num_workers=args.workers)
        
        # Analyze and plot
        sweep.plot_stability_region(save_path=args.save)
        
        if args.save:
            sweep.save_results(args.save + '.pkl')
    
    elif args.mode == 'test':
        # Quick test mode
        print("Running quick stability test...")
        sim = SLATOLSimulator(gui=False)
        sim.initialize_robot()
        
        # Test with medium parameters
        sim.leg_mass_ratio = 0.2
        sim.wind_ratio = 0.1
        sim.robot.update_mass_distribution(sim.leg_mass_ratio)
        
        sim.run_simulation(duration=args.duration)
        
        if sim.crash_detected:
            print("TEST FAILED: Robot crashed")
        else:
            print("TEST PASSED: Robot stable")
        
        sim.close()
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()