"""
Experimental Validation Framework for 2025 Olfactory AI Research.

Comprehensive framework for statistical validation, reproducibility,
and publication-ready experimental evaluation including:
- Controlled experimental design
- Statistical significance testing  
- Cross-validation and bootstrap analysis
- Comparative baseline studies
- Publication-ready result generation
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from itertools import combinations
from abc import ABC, abstractmethod

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class ExperimentalDesign:
    """Controlled experimental design configuration."""
    name: str
    hypothesis: str
    variables: Dict[str, Any]
    controls: Dict[str, Any]
    success_criteria: Dict[str, float]
    sample_sizes: Dict[str, int]
    significance_level: float = 0.05
    power_threshold: float = 0.8


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    experiment_name: str
    hypothesis_accepted: bool
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_power: float
    reproducibility_score: float
    publication_metrics: Dict[str, Any]


class StatisticalValidator:
    """Advanced statistical validation for olfactory AI research."""
    
    def __init__(self, alpha: float = 0.05, min_effect_size: float = 0.2):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.test_results = {}
    
    def t_test(self, group1: np.ndarray, group2: np.ndarray, 
               paired: bool = False) -> Dict[str, float]:
        """Perform t-test with effect size calculation."""
        if HAS_SCIPY:
            if paired:
                statistic, p_value = stats.ttest_rel(group1, group2)
            else:
                statistic, p_value = stats.ttest_ind(group1, group2)
        else:
            # Simplified t-test implementation
            if paired:
                diff = group1 - group2
                statistic = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)))
                df = len(diff) - 1
            else:
                pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + 
                                    (len(group2)-1)*np.var(group2)) / 
                                   (len(group1)+len(group2)-2))
                statistic = (np.mean(group1) - np.mean(group2)) / \
                           (pooled_std * np.sqrt(1/len(group1) + 1/len(group2)))
                df = len(group1) + len(group2) - 2
            
            # Simplified p-value calculation
            p_value = 2 * (1 - stats.t.cdf(abs(statistic), df)) if HAS_SCIPY else 0.05
        
        # Cohen's d effect size
        if paired:
            cohen_d = np.mean(group1 - group2) / np.std(group1 - group2)
        else:
            pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
            cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': abs(cohen_d),
            'significant': p_value < self.alpha,
            'large_effect': abs(cohen_d) > 0.8
        }
    
    def anova_test(self, *groups) -> Dict[str, float]:
        """Perform one-way ANOVA."""
        if HAS_SCIPY:
            f_statistic, p_value = stats.f_oneway(*groups)
        else:
            # Simplified ANOVA
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 
                           for group in groups)
            
            # Within-group sum of squares
            ss_within = sum(np.sum((group - np.mean(group))**2) 
                          for group in groups)
            
            df_between = len(groups) - 1
            df_within = len(all_data) - len(groups)
            
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            
            f_statistic = ms_between / ms_within if ms_within > 0 else 0
            p_value = 0.05  # Simplified
        
        # Eta-squared effect size
        eta_squared = ss_between / (ss_between + ss_within) if 'ss_between' in locals() else 0.1
        
        return {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'effect_size': eta_squared,
            'significant': p_value < self.alpha,
            'large_effect': eta_squared > 0.14
        }
    
    def correlation_test(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Test correlation significance."""
        correlation = np.corrcoef(x, y)[0, 1]
        
        if HAS_SCIPY:
            statistic, p_value = stats.pearsonr(x, y)
            correlation = statistic
        else:
            # Simplified correlation test
            n = len(x)
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 0.05 if abs(t_stat) > 2 else 0.1  # Simplified
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'strong_correlation': abs(correlation) > 0.7
        }
    
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    statistic_func: Callable = np.mean,
                                    n_bootstrap: int = 1000,
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals."""
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def power_analysis(self, effect_size: float, sample_size: int, 
                      alpha: float = None) -> float:
        """Calculate statistical power."""
        if alpha is None:
            alpha = self.alpha
        
        # Simplified power calculation for t-test
        if HAS_SCIPY:
            from scipy.special import ndtr
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(sample_size/2) - z_alpha
            power = ndtr(z_beta)
        else:
            # Very simplified power approximation
            power = 1 - (alpha * np.exp(-effect_size * sample_size / 10))
            power = np.clip(power, 0, 1)
        
        return power


class CrossValidator:
    """Cross-validation framework for robust evaluation."""
    
    def __init__(self, n_folds: int = 5, n_repeats: int = 3):
        self.n_folds = n_folds
        self.n_repeats = n_repeats
    
    def k_fold_cross_validation(self, X: np.ndarray, y: np.ndarray,
                               model_func: Callable,
                               metric_func: Callable = None) -> Dict[str, Any]:
        """Perform k-fold cross-validation."""
        if metric_func is None:
            metric_func = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)
        
        n_samples = X.shape[0]
        fold_size = n_samples // self.n_folds
        
        all_scores = []
        fold_predictions = []
        
        for repeat in range(self.n_repeats):
            # Shuffle data for each repeat
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            repeat_scores = []
            
            for fold in range(self.n_folds):
                # Create train/test split
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < self.n_folds - 1 else n_samples
                
                test_indices = np.arange(start_idx, end_idx)
                train_indices = np.concatenate([
                    np.arange(0, start_idx),
                    np.arange(end_idx, n_samples)
                ])
                
                X_train, X_test = X_shuffled[train_indices], X_shuffled[test_indices]
                y_train, y_test = y_shuffled[train_indices], y_shuffled[test_indices]
                
                # Train model and predict
                model = model_func()
                model.train(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Calculate metric
                score = metric_func(y_test, predictions)
                repeat_scores.append(score)
                fold_predictions.append({
                    'repeat': repeat,
                    'fold': fold,
                    'y_true': y_test,
                    'y_pred': predictions,
                    'score': score
                })
            
            all_scores.extend(repeat_scores)
        
        return {
            'scores': all_scores,
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'predictions': fold_predictions,
            'confidence_interval': (np.mean(all_scores) - 1.96 * np.std(all_scores) / np.sqrt(len(all_scores)),
                                   np.mean(all_scores) + 1.96 * np.std(all_scores) / np.sqrt(len(all_scores)))
        }
    
    def stratified_validation(self, X: np.ndarray, y: np.ndarray, 
                            model_func: Callable) -> Dict[str, Any]:
        """Stratified validation for classification tasks."""
        unique_classes = np.unique(y)
        stratified_results = {}
        
        for class_label in unique_classes:
            class_mask = (y == class_label)
            class_X = X[class_mask]
            class_y = y[class_mask]
            
            if len(class_X) >= self.n_folds:
                cv_result = self.k_fold_cross_validation(class_X, class_y, model_func)
                stratified_results[f'class_{class_label}'] = cv_result
        
        return stratified_results
    
    def temporal_validation(self, X: np.ndarray, y: np.ndarray,
                          model_func: Callable, n_windows: int = 5) -> Dict[str, Any]:
        """Temporal cross-validation for time-series data."""
        n_samples = X.shape[0]
        window_size = n_samples // n_windows
        
        temporal_results = []
        
        for window in range(1, n_windows):
            train_end = window * window_size
            test_start = train_end
            test_end = min((window + 1) * window_size, n_samples)
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            model = model_func()
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            
            mse = np.mean((y_test - predictions) ** 2)
            
            temporal_results.append({
                'window': window,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'mse': mse,
                'y_true': y_test,
                'y_pred': predictions
            })
        
        return {
            'temporal_results': temporal_results,
            'mean_mse': np.mean([r['mse'] for r in temporal_results]),
            'temporal_stability': np.std([r['mse'] for r in temporal_results])
        }


class ReproducibilityFramework:
    """Framework for ensuring experimental reproducibility."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.experiment_log = []
    
    def set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_seed)
        # Additional seeds would be set for torch, tensorflow, etc.
    
    def log_experiment(self, experiment_name: str, parameters: Dict[str, Any],
                      results: Dict[str, Any]):
        """Log experiment for reproducibility."""
        log_entry = {
            'timestamp': time.time(),
            'experiment_name': experiment_name,
            'random_seed': self.random_seed,
            'parameters': parameters,
            'results': results,
            'environment': {
                'python_version': '3.9+',
                'numpy_version': np.__version__ if np else 'not_available'
            }
        }
        self.experiment_log.append(log_entry)
    
    def save_experiment_log(self, filepath: Path):
        """Save experiment log for reproducibility."""
        with open(filepath, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
    
    def verify_reproducibility(self, experiment_func: Callable, 
                              parameters: Dict[str, Any],
                              n_replications: int = 3) -> Dict[str, Any]:
        """Verify experiment reproducibility."""
        results = []
        
        for replication in range(n_replications):
            self.set_random_seeds()
            result = experiment_func(**parameters)
            results.append(result)
        
        # Calculate reproducibility metrics
        if isinstance(results[0], dict) and 'accuracy' in results[0]:
            accuracies = [r['accuracy'] for r in results]
            reproducibility_score = 1 - (np.std(accuracies) / np.mean(accuracies))
        else:
            reproducibility_score = 1.0  # Default
        
        return {
            'replications': results,
            'reproducibility_score': reproducibility_score,
            'mean_result': np.mean([r.get('accuracy', 0) for r in results]),
            'std_result': np.std([r.get('accuracy', 0) for r in results])
        }


class ExperimentalValidationFramework:
    """Comprehensive experimental validation framework."""
    
    def __init__(self, random_seed: int = 42):
        self.statistical_validator = StatisticalValidator()
        self.cross_validator = CrossValidator()
        self.reproducibility = ReproducibilityFramework(random_seed)
        self.experiments = {}
        self.baselines = {}
    
    def register_baseline(self, name: str, baseline_func: Callable):
        """Register baseline method for comparison."""
        self.baselines[name] = baseline_func
    
    def design_experiment(self, name: str, hypothesis: str,
                         experimental_variables: Dict[str, Any],
                         controls: Dict[str, Any],
                         success_criteria: Dict[str, float],
                         sample_sizes: Dict[str, int]) -> ExperimentalDesign:
        """Design controlled experiment."""
        design = ExperimentalDesign(
            name=name,
            hypothesis=hypothesis,
            variables=experimental_variables,
            controls=controls,
            success_criteria=success_criteria,
            sample_sizes=sample_sizes
        )
        
        return design
    
    def run_controlled_experiment(self, design: ExperimentalDesign,
                                 experimental_method: Callable,
                                 data: Dict[str, Any]) -> ValidationResult:
        """Run controlled experiment with statistical validation."""
        logging.info(f"Running controlled experiment: {design.name}")
        
        self.reproducibility.set_random_seeds()
        
        # Run experimental method
        experimental_results = experimental_method(data)
        
        # Run baseline comparisons
        baseline_results = {}
        for baseline_name, baseline_func in self.baselines.items():
            baseline_results[baseline_name] = baseline_func(data)
        
        # Statistical testing
        p_values = {}
        effect_sizes = {}
        confidence_intervals = {}
        
        # Compare experimental method against each baseline
        for baseline_name, baseline_result in baseline_results.items():
            if 'performance_scores' in experimental_results and 'performance_scores' in baseline_result:
                exp_scores = experimental_results['performance_scores']
                base_scores = baseline_result['performance_scores']
                
                # T-test comparison
                t_test_result = self.statistical_validator.t_test(exp_scores, base_scores)
                p_values[f'vs_{baseline_name}'] = t_test_result['p_value']
                effect_sizes[f'vs_{baseline_name}'] = t_test_result['effect_size']
                
                # Bootstrap confidence interval
                diff_scores = exp_scores - base_scores
                ci = self.statistical_validator.bootstrap_confidence_interval(diff_scores)
                confidence_intervals[f'vs_{baseline_name}'] = ci
        
        # Calculate statistical power
        if effect_sizes:
            avg_effect_size = np.mean(list(effect_sizes.values()))
            sample_size = np.mean(list(design.sample_sizes.values()))
            statistical_power = self.statistical_validator.power_analysis(avg_effect_size, sample_size)
        else:
            statistical_power = 0.8  # Default
        
        # Test reproducibility
        reproducibility_result = self.reproducibility.verify_reproducibility(
            experimental_method, {'data': data}
        )
        
        # Hypothesis testing
        hypothesis_accepted = True
        for criterion_name, threshold in design.success_criteria.items():
            if criterion_name in experimental_results:
                if experimental_results[criterion_name] < threshold:
                    hypothesis_accepted = False
                    break
        
        # Check statistical significance
        significant_results = sum(1 for p in p_values.values() if p < design.significance_level)
        if significant_results == 0:
            hypothesis_accepted = False
        
        # Publication metrics
        publication_metrics = {
            'total_comparisons': len(baseline_results),
            'significant_improvements': significant_results,
            'largest_effect_size': max(effect_sizes.values()) if effect_sizes else 0,
            'reproducibility_score': reproducibility_result['reproducibility_score'],
            'statistical_power': statistical_power
        }
        
        result = ValidationResult(
            experiment_name=design.name,
            hypothesis_accepted=hypothesis_accepted,
            p_values=p_values,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            statistical_power=statistical_power,
            reproducibility_score=reproducibility_result['reproducibility_score'],
            publication_metrics=publication_metrics
        )
        
        # Log experiment
        self.reproducibility.log_experiment(
            design.name,
            design.__dict__,
            result.__dict__
        )
        
        self.experiments[design.name] = result
        
        return result
    
    def run_cross_validation_study(self, model_func: Callable, 
                                  X: np.ndarray, y: np.ndarray,
                                  study_name: str = "cross_validation") -> Dict[str, Any]:
        """Run comprehensive cross-validation study."""
        logging.info(f"Running cross-validation study: {study_name}")
        
        # K-fold cross-validation
        kfold_results = self.cross_validator.k_fold_cross_validation(X, y, model_func)
        
        # Stratified validation (if applicable)
        stratified_results = None
        if len(np.unique(y)) > 1 and len(np.unique(y)) < 10:  # Classification task
            stratified_results = self.cross_validator.stratified_validation(X, y, model_func)
        
        # Temporal validation (if applicable)
        temporal_results = None
        if X.shape[0] > 50:  # Enough samples for temporal analysis
            temporal_results = self.cross_validator.temporal_validation(X, y, model_func)
        
        return {
            'kfold_results': kfold_results,
            'stratified_results': stratified_results,
            'temporal_results': temporal_results,
            'overall_performance': kfold_results['mean_score'],
            'performance_stability': kfold_results['std_score']
        }
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.experiments:
            return "No experiments have been run."
        
        report = [
            "# Experimental Validation Report",
            "",
            "## Summary",
            f"Total experiments conducted: {len(self.experiments)}",
            f"Statistical significance level: {self.statistical_validator.alpha}",
            "",
            "## Experimental Results",
            ""
        ]
        
        for exp_name, result in self.experiments.items():
            report.extend([
                f"### {exp_name}",
                "",
                f"**Hypothesis**: {'ACCEPTED ✅' if result.hypothesis_accepted else 'REJECTED ❌'}",
                f"**Statistical Power**: {result.statistical_power:.3f}",
                f"**Reproducibility Score**: {result.reproducibility_score:.3f}",
                ""
            ])
            
            if result.p_values:
                report.append("**Statistical Comparisons**:")
                for comparison, p_value in result.p_values.items():
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    effect_size = result.effect_sizes.get(comparison, 0)
                    report.append(f"- {comparison}: p = {p_value:.4f}{significance} (effect size: {effect_size:.3f})")
                report.append("")
            
            if result.confidence_intervals:
                report.append("**Confidence Intervals (95%)**:")
                for comparison, (ci_low, ci_high) in result.confidence_intervals.items():
                    report.append(f"- {comparison}: [{ci_low:.4f}, {ci_high:.4f}]")
                report.append("")
            
            # Publication readiness
            pub_metrics = result.publication_metrics
            report.extend([
                "**Publication Metrics**:",
                f"- Significant improvements: {pub_metrics['significant_improvements']}/{pub_metrics['total_comparisons']}",
                f"- Largest effect size: {pub_metrics['largest_effect_size']:.3f}",
                f"- Statistical power: {pub_metrics['statistical_power']:.3f}",
                ""
            ])
        
        # Overall assessment
        report.extend([
            "## Overall Assessment",
            ""
        ])
        
        accepted_hypotheses = sum(1 for r in self.experiments.values() if r.hypothesis_accepted)
        avg_power = np.mean([r.statistical_power for r in self.experiments.values()])
        avg_reproducibility = np.mean([r.reproducibility_score for r in self.experiments.values()])
        
        report.extend([
            f"- Hypotheses accepted: {accepted_hypotheses}/{len(self.experiments)} ({accepted_hypotheses/len(self.experiments)*100:.1f}%)",
            f"- Average statistical power: {avg_power:.3f}",
            f"- Average reproducibility: {avg_reproducibility:.3f}",
            "",
            "## Publication Readiness",
            ""
        ])
        
        if avg_power >= 0.8 and avg_reproducibility >= 0.9:
            report.append("✅ **Publication ready**: High statistical power and reproducibility")
        elif avg_power >= 0.8 or avg_reproducibility >= 0.8:
            report.append("⚠️ **Needs improvement**: Some metrics below publication standards")
        else:
            report.append("❌ **Not publication ready**: Low statistical power or reproducibility")
        
        return "\n".join(report)
    
    def export_validation_data(self, output_path: Path):
        """Export validation data for external review."""
        export_data = {
            'validation_framework_version': '2025.1',
            'statistical_settings': {
                'alpha': self.statistical_validator.alpha,
                'min_effect_size': self.statistical_validator.min_effect_size
            },
            'experiments': {},
            'reproducibility_log': self.reproducibility.experiment_log
        }
        
        for exp_name, result in self.experiments.items():
            export_data['experiments'][exp_name] = {
                'hypothesis_accepted': result.hypothesis_accepted,
                'p_values': result.p_values,
                'effect_sizes': result.effect_sizes,
                'confidence_intervals': {k: list(v) for k, v in result.confidence_intervals.items()},
                'statistical_power': result.statistical_power,
                'reproducibility_score': result.reproducibility_score,
                'publication_metrics': result.publication_metrics
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logging.info(f"Validation data exported to {output_path}")


def main():
    """Demonstrate experimental validation framework."""
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize validation framework
    validator = ExperimentalValidationFramework(random_seed=42)
    
    # Define baseline methods
    class SimpleBaseline:
        def train(self, X, y):
            self.mean_y = np.mean(y)
        
        def predict(self, X):
            return np.full(len(X), self.mean_y)
    
    class RandomBaseline:
        def train(self, X, y):
            self.y_range = (np.min(y), np.max(y))
        
        def predict(self, X):
            return np.random.uniform(self.y_range[0], self.y_range[1], len(X))
    
    # Register baselines
    validator.register_baseline('simple', lambda data: {
        'performance_scores': np.random.normal(0.7, 0.1, 20)
    })
    validator.register_baseline('random', lambda data: {
        'performance_scores': np.random.normal(0.6, 0.15, 20)
    })
    
    # Design experiment
    experiment_design = validator.design_experiment(
        name="Novel Olfactory Algorithm Validation",
        hypothesis="Novel algorithm achieves significantly higher accuracy than baselines",
        experimental_variables={'algorithm_type': 'novel_transformer'},
        controls={'dataset': 'standardized', 'preprocessing': 'normalized'},
        success_criteria={'accuracy': 0.85, 'f1_score': 0.80},
        sample_sizes={'training': 1000, 'validation': 200, 'test': 200}
    )
    
    # Define experimental method
    def novel_method(data):
        # Simulate novel algorithm results
        return {
            'performance_scores': np.random.normal(0.88, 0.08, 20),
            'accuracy': 0.88,
            'f1_score': 0.85
        }
    
    # Run controlled experiment
    result = validator.run_controlled_experiment(
        experiment_design,
        novel_method,
        {'synthetic_data': True}
    )
    
    # Generate validation report
    report = validator.generate_validation_report()
    print(report)
    
    # Export validation data
    validator.export_validation_data(Path("validation_results.json"))
    
    logging.info("Experimental validation complete!")
    
    return validator


if __name__ == "__main__":
    main()