"""Command-line interface for Olfactory Transformer."""

from typing import List, Optional
import logging
from pathlib import Path
import json

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel

from .core.model import OlfactoryTransformer
from .core.tokenizer import MoleculeTokenizer
from .core.config import OlfactoryConfig
from .sensors.enose import ENoseInterface
from .design.inverse import ScentDesigner
from .evaluation.metrics import PerceptualEvaluator
from .training.trainer import OlfactoryTrainer, OlfactoryDataset, TrainingArguments

app = typer.Typer(help="Olfactory Transformer CLI - Foundation model for computational olfaction")
console = Console()

# Setup logging
logging.basicConfig(level=logging.INFO)


@app.command()
def predict(
    smiles: str = typer.Argument(..., help="SMILES string of molecule to analyze"),
    model_path: str = typer.Option("olfactory-base-v1", help="Path to pre-trained model"),
    output_format: str = typer.Option("text", help="Output format: text, json"),
    detailed: bool = typer.Option(False, help="Show detailed analysis"),
):
    """Predict scent properties from molecular structure."""
    console.print(f"[blue]Analyzing molecule:[/blue] {smiles}")
    
    try:
        # Load model and tokenizer
        with console.status("Loading model..."):
            model = OlfactoryTransformer.from_pretrained(model_path)
            tokenizer = MoleculeTokenizer.from_pretrained(model_path)
        
        # Make prediction
        with console.status("Making prediction..."):
            prediction = model.predict_scent(smiles, tokenizer)
        
        # Display results
        if output_format == "json":
            console.print(json.dumps(prediction.to_dict(), indent=2))
        else:
            _display_prediction(prediction, detailed)
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def sensor_stream(
    port: str = typer.Option("/dev/ttyUSB0", help="Serial port for sensor array"),
    sensors: List[str] = typer.Option(["TGS2600", "TGS2602"], help="Sensor types"),
    duration: float = typer.Option(30.0, help="Streaming duration in seconds"),
    model_path: str = typer.Option("olfactory-base-v1", help="Path to pre-trained model"),
):
    """Stream real-time scent detection from sensor array."""
    console.print("[blue]Starting sensor streaming...[/blue]")
    
    try:
        # Setup sensor interface
        enose = ENoseInterface(port=port, sensors=sensors)
        
        # Load model
        model = OlfactoryTransformer.from_pretrained(model_path)
        
        # Stream predictions
        with enose.stream(duration=duration) as sensor_stream:
            for reading in sensor_stream:
                prediction = model.predict_from_sensors(reading)
                
                console.print(f"[green]Detected:[/green] {prediction.primary_notes[0] if prediction.primary_notes else 'unknown'} "
                            f"(confidence: {prediction.confidence:.2%})")
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def design(
    notes: List[str] = typer.Option(..., help="Target scent notes"),
    intensity: float = typer.Option(7.0, help="Target intensity (1-10)"),
    n_candidates: int = typer.Option(10, help="Number of molecule candidates"),
    model_path: str = typer.Option("olfactory-base-v1", help="Path to pre-trained model"),
    output_file: Optional[str] = typer.Option(None, help="Save results to file"),
):
    """Design molecules for target scent profile."""
    console.print(f"[blue]Designing molecules for:[/blue] {', '.join(notes)}")
    
    try:
        # Load model
        model = OlfactoryTransformer.from_pretrained(model_path)
        
        # Create designer
        designer = ScentDesigner(model)
        
        # Design molecules
        target_profile = {
            "notes": notes,
            "intensity": intensity,
            "longevity": "high",
            "character": "sophisticated"
        }
        
        with console.status("Generating molecular candidates..."):
            candidates = designer.design_molecules(target_profile, n_candidates=n_candidates)
        
        # Display results
        _display_molecule_candidates(candidates)
        
        # Save if requested
        if output_file:
            designer.save_design_session(candidates, output_file)
            console.print(f"[green]Results saved to:[/green] {output_file}")
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def train(
    molecules: str = typer.Argument(..., help="Path to molecule data (CSV/SDF)"),
    descriptions: str = typer.Argument(..., help="Path to scent descriptions (JSON/CSV)"),
    output_dir: str = typer.Option("./model_output", help="Output directory"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(1e-5, help="Learning rate"),
):
    """Train olfactory transformer model."""
    console.print("[blue]Starting model training...[/blue]")
    
    try:
        # Create dataset
        tokenizer = MoleculeTokenizer()
        dataset = OlfactoryDataset(molecules, descriptions, tokenizer=tokenizer)
        train_dataset, eval_dataset = dataset.split(0.1)
        
        # Create model
        config = OlfactoryConfig()
        model = OlfactoryTransformer(config)
        
        # Setup training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        
        # Create trainer
        trainer = OlfactoryTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
        )
        
        # Train model
        trainer.train()
        
        console.print(f"[green]Training completed! Model saved to:[/green] {output_dir}")
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    test_data: str = typer.Argument(..., help="Path to test dataset"),
    panel_data: Optional[str] = typer.Option(None, help="Path to human panel data"),
    output_file: Optional[str] = typer.Option(None, help="Save evaluation report"),
):
    """Evaluate model performance."""
    console.print("[blue]Evaluating model...[/blue]")
    
    try:
        # Load model
        model = OlfactoryTransformer.from_pretrained(model_path)
        
        # Create evaluator
        evaluator = PerceptualEvaluator(model, panel_data)
        
        # Run evaluation
        with console.status("Running evaluation..."):
            results = evaluator.evaluate_model(test_data)
        
        # Display results
        _display_evaluation_results(results)
        
        # Generate report
        if output_file:
            report = evaluator.generate_evaluation_report(results)
            with open(output_file, 'w') as f:
                f.write(report)
            console.print(f"[green]Report saved to:[/green] {output_file}")
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def calibrate_sensors(
    port: str = typer.Option("/dev/ttyUSB0", help="Serial port"),
    compounds: List[str] = typer.Option(["ethanol", "acetone"], help="Reference compounds"),
    concentrations: List[float] = typer.Option([10, 50, 100], help="Concentrations (ppm)"),
    output_file: str = typer.Option("calibration.json", help="Calibration output file"),
):
    """Calibrate sensor array with reference compounds."""
    console.print("[blue]Starting sensor calibration...[/blue]")
    
    try:
        # Setup sensor interface
        enose = ENoseInterface(port=port)
        
        # Run calibration
        with Progress() as progress:
            task = progress.add_task("Calibrating...", total=len(compounds) * len(concentrations))
            
            calibration_data = enose.calibrate(compounds, concentrations)
            progress.update(task, completed=len(compounds) * len(concentrations))
        
        # Save calibration
        enose.save_calibration(output_file)
        
        console.print(f"[green]Calibration completed! Data saved to:[/green] {output_file}")
    
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _display_prediction(prediction, detailed: bool = False):
    """Display scent prediction results."""
    # Main results panel
    content = f"[bold]Primary Notes:[/bold] {', '.join(prediction.primary_notes)}\n"
    content += f"[bold]Intensity:[/bold] {prediction.intensity:.1f}/10\n"
    content += f"[bold]Confidence:[/bold] {prediction.confidence:.1%}\n"
    content += f"[bold]Chemical Family:[/bold] {prediction.chemical_family}"
    
    if detailed:
        content += f"\n[bold]IFRA Category:[/bold] {prediction.ifra_category}"
        content += f"\n[bold]Similar Perfumes:[/bold] {', '.join(prediction.similar_perfumes)}"
    
    console.print(Panel(content, title="Scent Prediction", border_style="green"))


def _display_molecule_candidates(candidates):
    """Display molecule design candidates."""
    table = Table(title="Molecule Candidates")
    table.add_column("Rank", style="cyan")
    table.add_column("SMILES", style="magenta")
    table.add_column("Match %", style="green")
    table.add_column("MW", style="blue")
    table.add_column("LogP", style="yellow")
    table.add_column("SA Score", style="red")
    
    for i, candidate in enumerate(candidates, 1):
        table.add_row(
            str(i),
            candidate.smiles[:50] + "..." if len(candidate.smiles) > 50 else candidate.smiles,
            f"{candidate.profile_match:.1%}",
            f"{candidate.molecular_weight:.1f}",
            f"{candidate.logp:.2f}",
            f"{candidate.sa_score:.2f}"
        )
    
    console.print(table)


def _display_evaluation_results(results):
    """Display evaluation results."""
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Benchmark", style="blue")
    table.add_column("Status", style="bold")
    
    for metric_name, result in results.items():
        status = "✅ PASS" if result.benchmark and result.score > result.benchmark else "❌ FAIL"
        if not result.benchmark:
            status = "ℹ️ N/A"
        
        table.add_row(
            metric_name.replace('_', ' ').title(),
            f"{result.score:.3f}",
            f"{result.benchmark:.3f}" if result.benchmark else "N/A",
            status
        )
    
    console.print(table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()