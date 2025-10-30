"""
Main Attack Script - Works with the FL Output
Save this as: src/attacks/run_attacks.py
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import gradient metrics
from quality_metrics import (
    GradientQualityMetrics, 
    AttackSuccessMetrics, 
    load_gradients_from_fl_output,
    get_round_info
)

def main():
    print("=" * 70)
    print("FEDERATED LEARNING GRADIENT ATTACK PIPELINE")
    print("=" * 70)
    
    # Step 1: Load file
    gradients_dir = Path("src/fl_simulation")  # Where FL files are saved
    print(f"\n📂 Loading gradients from: {gradients_dir}")
    
    # Find all tensor files
    tensor_files = list(gradients_dir.glob("*_tensors.pt"))
    if not tensor_files:
        print(f"❌ No gradient files found!")
        print(f"💡 Run the FL training first to generate files")
        print(f"   Expected files: {gradients_dir}/*_tensors.pt")
        return
    
    print(f"✅ Found {len(tensor_files)} gradient files")
    for f in sorted(tensor_files):
        print(f"   📄 {f.name}")
    
    # Use the latest round
    latest_file = sorted(tensor_files)[-1]
    print(f"\n🎯 Analyzing: {latest_file.name}")
    
    # Step 2: Load the data
    try:
        tensor_data = torch.load(latest_file, map_location='cpu')
        
        # Check structure
        print(f"\n📊 File structure:")
        print(f"   Round: {latest_file.stem}")
        print(f"   Clients with gradients: {list(tensor_data['raw_gradients'].keys())}")
        
        # Get first client's gradients
        client_id = list(tensor_data['raw_gradients'].keys())[0]
        client_grads = tensor_data['raw_gradients'][client_id]
        
        print(f"   Client {client_id}: {len(client_grads['grads_per_step_raw'])} gradient steps")
        
        # For demonstration: compare first two steps as "original" vs "reconstructed"
        if len(client_grads['grads_per_step_raw']) >= 2:
            original_grads = client_grads['grads_per_step_raw'][0]
            reconstructed_grads = client_grads['grads_per_step_raw'][1]  # Simulate reconstruction
            
            print(f"   Using step 0 as 'original', step 1 as 'reconstructed' for demo")
            
        else:
            print(f"❌ Not enough gradient steps for analysis")
            return
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Step 3: Run gradient similarity analysis
    print(f"\n🎯 RUNNING GRADIENT SIMILARITY ATTACK...")
    attacker = GradientQualityMetrics()
    
    results = attacker.evaluate(original_grads, reconstructed_grads)
    leakage_score = AttackSuccessMetrics.privacy_leakage_score(original_grads, reconstructed_grads)
    risk_level, risk_desc = AttackSuccessMetrics.classify_privacy_risk(results['cosine_similarity'])
    
    # Step 4: Display results
    print(f"\n{'='*70}")
    print(f"ATTACK RESULTS - Client {client_id}:")
    print(f"{'='*70}")
    print(f"Cosine Similarity:     {results['cosine_similarity']:.4f}")
    print(f"L2 Distance:           {results['l2_distance']:.4f}")
    print(f"Directional Alignment: {results['directional_alignment']:.4f}")
    print(f"Gradient Norm Ratio:   {results['gradient_norm_ratio']:.4f}")
    print(f"Parameter Correlation: {results['parameter_correlation']:.4f}")
    print(f"Privacy Leakage Score: {leakage_score:.4f}")
    
    print(f"\n💡 SECURITY ASSESSMENT:")
    print(f"   Risk Level: {risk_level}")
    print(f"   Description: {risk_desc}")
    
    # Step 5: Save results
    results_dir = Path("reports/attacks")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    gradients_dir = Path("src/attacks/gradients") 
    gradients_dir.mkdir(parents=True, exist_ok=True)  # Creates src/attacks/gradients/

    results_data = {
        'timestamp': datetime.now().isoformat(),
        'source_file': latest_file.name,
        'client_id': client_id,
        'metrics': results,
        'privacy_assessment': {
            'leakage_score': leakage_score,
            'risk_level': risk_level,
            'description': risk_desc
        }
    }
    
    results_path = results_dir / f"attack_results_{latest_file.stem}.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n✅ Saved results to: {results_path}")
    print(f"\n{'='*70}")
    print(f"✅ GRADIENT ATTACK ANALYSIS COMPLETE!")
    print(f"{'='*70}")

    # Step 6: Create visualization
    print(f"\n🎨 Creating visualization...")
    
    # Plot the metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metric values for bar chart
    metric_names = ['Cosine Similarity', 'L2 Distance', 'Directional Alignment', 'Leakage Score']
    metric_values = [
        results['cosine_similarity'],
        results['l2_distance'],
        results['directional_alignment'], 
        leakage_score
    ]
    colors = ['blue', 'red', 'green', 'orange']
    
    # Bar chart
    bars = axes[0, 0].bar(metric_names, metric_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Gradient Similarity Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Risk assessment
    risk_levels = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    risk_scores = [0.0, 0.2, 0.4, 0.6, 0.8]
    current_risk_index = risk_scores.index(next(x for x in risk_scores if x <= results['cosine_similarity']))
    
    axes[0, 1].barh(risk_levels, [1]*5, color='lightgray', alpha=0.3)
    axes[0, 1].barh(risk_levels[:current_risk_index+1], [1]*(current_risk_index+1), 
                    color='red', alpha=0.7)
    axes[0, 1].set_title('Privacy Risk Level')
    axes[0, 1].set_xlabel('Risk Progression')
    
    # Layer-wise similarities
    if results['layer_wise_similarity']:
        layers = list(results['layer_wise_similarity'].keys())
        layer_cos_sim = [results['layer_wise_similarity'][l]['cosine_similarity'] for l in layers]
        # Shorten layer names for display
        short_layers = [l.split('.')[-1][:15] for l in layers]
        
        axes[1, 0].barh(short_layers, layer_cos_sim, alpha=0.7)
        axes[1, 0].set_title('Layer-wise Cosine Similarity')
        axes[1, 0].set_xlabel('Cosine Similarity')
        axes[1, 0].axvline(x=0.7, color='r', linestyle='--', alpha=0.5, label='Danger Threshold')
    
    # Empty subplot for text summary
    axes[1, 1].axis('off')
    summary_text = f"""
    Attack Summary:
    
    File: {latest_file.name}
    Client: {client_id}
    Risk Level: {risk_level}
    
    Key Findings:
    • Cosine Similarity: {results['cosine_similarity']:.3f}
    • Privacy Leakage: {leakage_score:.1%}
    • Status: {risk_desc}
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Gradient Attack Analysis - {risk_level} Risk', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    viz_path = results_dir / f"attack_visualization_{latest_file.stem}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {viz_path}")

if __name__ == "__main__":
    main()