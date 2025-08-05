"""Basic usage examples for Olfactory Transformer."""

import torch
from olfactory_transformer import (
    OlfactoryTransformer,
    MoleculeTokenizer,
    ENoseInterface,
    ScentDesigner
)
from olfactory_transformer.core.config import OlfactoryConfig


def example_scent_prediction():
    """Example: Predict scent from molecular structure."""
    print("🧬 Scent Prediction Example")
    print("=" * 40)
    
    # Load pre-trained model (in real use, would load from checkpoint)
    config = OlfactoryConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2)
    model = OlfactoryTransformer(config)
    model.eval()
    
    # Create tokenizer
    tokenizer = MoleculeTokenizer(vocab_size=1000)
    sample_molecules = [
        "CCO",                    # Ethanol
        "CC(C)CC1=CC=C(C=C1)C(C)C",  # Lily aldehyde
        "C1=CC=C(C=C1)C=O",      # Benzaldehyde
        "COC1=CC(=CC=C1O)C=O",   # Vanillin
    ]
    tokenizer.build_vocab_from_smiles(sample_molecules)
    
    # Test molecules
    test_molecules = [
        ("CCO", "Ethanol - simple alcohol"),
        ("CC(=O)OCC", "Ethyl acetate - fruity ester"),
        ("C1=CC=C(C=C1)C=O", "Benzaldehyde - almond-like"),
        ("COC1=CC(=CC=C1O)C=O", "Vanillin - vanilla scent"),
    ]
    
    for smiles, description in test_molecules:
        print(f"\nMolecule: {description}")
        print(f"SMILES: {smiles}")
        
        with torch.no_grad():
            prediction = model.predict_scent(smiles, tokenizer)
        
        print(f"Primary Notes: {', '.join(prediction.primary_notes)}")
        print(f"Intensity: {prediction.intensity:.1f}/10")
        print(f"Confidence: {prediction.confidence:.1%}")
        print(f"Chemical Family: {prediction.chemical_family}")


def example_sensor_integration():
    """Example: Real-time scent detection from sensors."""
    print("\n📡 Sensor Integration Example")
    print("=" * 40)
    
    # Setup electronic nose
    enose = ENoseInterface(
        sensors=["TGS2600", "TGS2602", "TGS2610", "TGS2620"],
        sampling_rate=2.0  # 2 Hz
    )
    
    # Load model
    config = OlfactoryConfig(sensor_channels=64)
    model = OlfactoryTransformer(config)
    model.eval()
    
    print("Starting sensor streaming for 5 seconds...")
    
    # Stream sensor data and make predictions
    detection_count = 0
    with enose.stream(duration=5.0) as sensor_stream:
        for reading in sensor_stream:
            with torch.no_grad():
                prediction = model.predict_from_sensors(reading)
            
            detection_count += 1
            print(f"Detection #{detection_count}:")
            print(f"  Detected: {prediction.primary_notes[0] if prediction.primary_notes else 'unknown'}")
            print(f"  Confidence: {prediction.confidence:.1%}")
            print(f"  Sensor values: {list(reading.gas_sensors.values())[:3]}...")
            
            if detection_count >= 3:  # Limit output for example
                break
    
    print(f"Processed {detection_count} sensor readings")


def example_molecular_design():
    """Example: Design molecules for target scent profile."""
    print("\n🧪 Molecular Design Example")
    print("=" * 40)
    
    # Load model
    config = OlfactoryConfig(vocab_size=500, hidden_size=32)
    model = OlfactoryTransformer(config)
    
    # Create designer
    designer = ScentDesigner(model)
    
    # Define target scent profile
    target_profile = {
        "notes": ["rose", "woody", "amber"],
        "intensity": 6.5,
        "longevity": "high",
        "character": "warm, sophisticated, romantic"
    }
    
    print("Designing molecules for target profile:")
    print(f"  Notes: {', '.join(target_profile['notes'])}")
    print(f"  Intensity: {target_profile['intensity']}/10")
    print(f"  Character: {target_profile['character']}")
    
    # Generate molecule candidates
    print("\nGenerating molecular candidates...")
    candidates = designer.design_molecules(
        target_profile,
        n_candidates=5,
        molecular_weight=(200, 350),
        logp=(2, 5)
    )
    
    print(f"\nGenerated {len(candidates)} candidate molecules:")
    print("-" * 80)
    
    for i, candidate in enumerate(candidates, 1):
        print(f"Candidate #{i}:")
        print(f"  SMILES: {candidate.smiles}")
        print(f"  Profile Match: {candidate.profile_match:.1%}")
        print(f"  Molecular Weight: {candidate.molecular_weight:.1f}")
        print(f"  LogP: {candidate.logp:.2f}")
        print(f"  SA Score: {candidate.sa_score:.2f}")
        print()
    
    # Create optimized formulation
    if candidates:
        print("Creating optimized formulation...")
        formulation = designer.optimize_formulation(candidates, target_profile)
        
        print("Recommended Formulation:")
        for smiles, percentage in formulation.items():
            print(f"  {smiles[:30]}... : {percentage:.1%}")


def example_batch_processing():
    """Example: Efficient batch processing of multiple molecules."""
    print("\n⚡ Batch Processing Example")
    print("=" * 40)
    
    from olfactory_transformer.utils.optimization import InferenceAccelerator
    
    # Setup model and accelerator
    config = OlfactoryConfig(vocab_size=200, hidden_size=32, num_hidden_layers=1)
    model = OlfactoryTransformer(config)
    model.eval()
    
    accelerator = InferenceAccelerator(
        model=model,
        max_batch_size=8,
        max_workers=2
    )
    
    # Create batch of molecules to process
    molecules = [
        "CCO", "CC(C)O", "CCC", "CCCC", "CCCCC",
        "C1=CC=CC=C1", "CC1=CC=CC=C1", "CCC1=CC=CC=C1",
        "CC(=O)OCC", "CC(=O)OCCC", "CCC(=O)OCC"
    ]
    
    print(f"Processing {len(molecules)} molecules in batches...")
    
    # Prepare batch data
    tokenizer = MoleculeTokenizer(vocab_size=200)
    tokenizer.build_vocab_from_smiles(molecules)
    
    batch_data = []
    for smiles in molecules:
        encoded = tokenizer.encode(smiles, max_length=20, padding=True)
        batch_data.append({
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"])
        })
    
    # Process with accelerator
    import time
    start_time = time.time()
    
    with accelerator:
        results = accelerator.predict_batch_sync(batch_data)
    
    end_time = time.time()
    
    print(f"Processed {len(results)} molecules in {end_time - start_time:.3f}s")
    
    # Show performance stats
    stats = accelerator.get_performance_stats()
    print(f"Batches processed: {stats['batches_processed']}")
    print(f"Average batch size: {stats['avg_batch_size']:.1f}")
    print(f"Throughput: {len(molecules)/(end_time - start_time):.1f} molecules/second")


def example_quality_control():
    """Example: Industrial quality control monitoring."""
    print("\n🏭 Quality Control Example")
    print("=" * 40)
    
    from olfactory_transformer.sensors.enose import ENoseArray, OlfactoryQualityControl
    
    # Setup multi-sensor array
    sensor_config = {
        "gas_sensors": ["TGS2600", "TGS2602", "TGS2610"],
        "environmental": ["BME680", "SHT31"],
        "spectrometer": "AS7265x"
    }
    
    sensor_array = ENoseArray(sensor_config)
    
    # Load model
    config = OlfactoryConfig()
    model = OlfactoryTransformer(config)
    
    # Setup QC system with reference profile
    reference_profile = {
        "target_notes": ["lavender", "fresh", "clean"],
        "intensity_range": [6.0, 8.0],
        "acceptable_compounds": ["linalool", "linalyl_acetate"],
        "forbidden_compounds": ["off_notes", "contaminants"]
    }
    
    qc_system = OlfactoryQualityControl(
        model=model,
        sensor_array=sensor_array,
        reference_profile=reference_profile,
        deviation_threshold=0.15  # 15% threshold
    )
    
    print("Starting quality control monitoring...")
    print("Reference profile: Lavender essential oil")
    print("Deviation threshold: 15%")
    
    # Monitor production batches
    batch_count = 0
    with sensor_array.acquire(duration=2.0) as data_stream:
        for batch_reading in qc_system.stream(batch_size=10):
            batch_count += 1
            print(f"\nBatch #{batch_count} Analysis:")
            print(f"  Deviation: {batch_reading.deviation:.1%}")
            
            if batch_reading.deviation > 0.15:
                print(f"  ⚠️  QUALITY ALERT!")
                print(f"  Outlier notes: {', '.join(batch_reading.outlier_notes)}")
                print(f"  Recommendation: {batch_reading.recommendation}")
            else:
                print(f"  ✅ Quality within acceptable range")
            
            if batch_count >= 2:  # Limit for example
                break
    
    print(f"\nProcessed {batch_count} production batches")


def example_performance_monitoring():
    """Example: Performance monitoring and optimization."""
    print("\n📊 Performance Monitoring Example")
    print("=" * 40)
    
    from olfactory_transformer.utils.monitoring import PerformanceMonitor, ResourceTracker
    
    # Setup monitoring
    performance_monitor = PerformanceMonitor(window_size=20)
    resource_tracker = ResourceTracker()
    
    # Setup model
    config = OlfactoryConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
    model = OlfactoryTransformer(config)
    model.eval()
    
    print("Running monitored inference tests...")
    
    # Run several inference operations with monitoring
    for i in range(5):
        # Create test input
        input_ids = torch.randint(0, 100, (4, 15))  # Batch of 4
        attention_mask = torch.ones(4, 15)
        
        with performance_monitor.time_inference(f"test_batch_{i}"):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get performance statistics
    perf_stats = performance_monitor.get_current_stats()
    print("\nPerformance Statistics:")
    print(f"  Total inferences: {perf_stats['total_inferences']}")
    print(f"  Average inference time: {perf_stats['recent_avg_inference_time']:.3f}s")
    print(f"  Average throughput: {perf_stats['recent_avg_throughput']:.1f} inferences/sec")
    print(f"  Peak memory usage: {perf_stats['recent_max_memory']:.1f} MB")
    
    # Get resource snapshot
    resource_snapshot = resource_tracker.get_resource_snapshot()
    print("\nSystem Resources:")
    print(f"  CPU usage: {resource_snapshot.cpu_percent:.1f}%")
    print(f"  Memory usage: {resource_snapshot.memory_percent:.1f}%")
    print(f"  Available memory: {resource_snapshot.memory_total_gb - resource_snapshot.memory_used_gb:.1f} GB")
    
    if resource_snapshot.gpu_utilization:
        print(f"  GPU utilization: {resource_snapshot.gpu_utilization:.1f}%")


def main():
    """Run all examples."""
    print("🌸 Olfactory Transformer - Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_scent_prediction()
        example_sensor_integration()
        example_molecular_design()
        example_batch_processing()
        example_quality_control()
        example_performance_monitoring()
        
        print("\n✅ All examples completed successfully!")
        print("\nFor more advanced usage, see:")
        print("  - Training: examples/training_example.py")
        print("  - Evaluation: examples/evaluation_example.py")
        print("  - Production: examples/production_deployment.py")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Note: Some examples require additional setup or hardware")


if __name__ == "__main__":
    main()