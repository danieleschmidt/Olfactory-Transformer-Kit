"""
Progressive Quality Gates - Autonomous quality enforcement that evolves.

This module implements progressive quality gates that automatically adapt 
and improve based on codebase evolution and learning from past failures.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import subprocess
import sys
import tempfile
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    EVOLVING = "evolving"


@dataclass
class QualityGateResult:
    """Result from a quality gate execution."""
    gate_name: str
    status: QualityGateStatus
    score: float  # 0.0 to 1.0
    execution_time: float
    details: Dict[str, Any]
    suggestions: List[str]
    auto_fixes_applied: List[str]
    evolution_data: Dict[str, Any]
    timestamp: datetime


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for progressive tracking."""
    code_coverage: float
    test_pass_rate: float
    performance_score: float
    security_score: float
    maintainability_index: float
    technical_debt_ratio: float
    reliability_score: float
    scalability_score: float


class AutomatedQualityValidator:
    """Validates quality and applies automated improvements."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.fix_patterns = {
            'import_sorting': self._fix_import_sorting,
            'code_formatting': self._fix_code_formatting,
            'type_hints': self._add_type_hints,
            'docstrings': self._add_docstrings,
            'security_fixes': self._apply_security_fixes,
        }
        
    def validate_and_fix(self, file_path: Path) -> QualityGateResult:
        """Validate file and apply automated fixes."""
        start_time = time.time()
        applied_fixes = []
        suggestions = []
        
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return QualityGateResult(
                    gate_name="file_validation",
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    execution_time=time.time() - start_time,
                    details={"error": f"File not found: {file_path}"},
                    suggestions=["Create the missing file"],
                    auto_fixes_applied=[],
                    evolution_data={},
                    timestamp=datetime.now()
                )
            
            # Read file content
            content = file_path.read_text()
            original_content = content
            
            # Apply automated fixes
            for fix_name, fix_func in self.fix_patterns.items():
                try:
                    fixed_content, fixes = fix_func(content)
                    if fixes:
                        applied_fixes.extend(fixes)
                        content = fixed_content
                except Exception as e:
                    suggestions.append(f"Manual fix needed for {fix_name}: {e}")
            
            # Write back if fixes were applied
            if applied_fixes and content != original_content:
                file_path.write_text(content)
                
            # Calculate quality score
            score = self._calculate_quality_score(file_path, content)
            
            status = QualityGateStatus.PASSED if score >= 0.8 else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="automated_validation",
                status=status,
                score=score,
                execution_time=time.time() - start_time,
                details={
                    "file_size": len(content),
                    "line_count": len(content.splitlines()),
                    "fixes_applied": len(applied_fixes)
                },
                suggestions=suggestions,
                auto_fixes_applied=applied_fixes,
                evolution_data={"original_score": score - len(applied_fixes) * 0.05},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            return QualityGateResult(
                gate_name="automated_validation",
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"error": str(e)},
                suggestions=["Manual review required"],
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
    
    def _fix_import_sorting(self, content: str) -> Tuple[str, List[str]]:
        """Apply import sorting fixes."""
        fixes = []
        lines = content.splitlines()
        import_lines = []
        other_lines = []
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(line)
            else:
                other_lines.append(line)
        
        if import_lines:
            sorted_imports = sorted(import_lines)
            if sorted_imports != import_lines:
                fixes.append("Sorted imports alphabetically")
                return '\n'.join(sorted_imports + other_lines), fixes
        
        return content, fixes
    
    def _fix_code_formatting(self, content: str) -> Tuple[str, List[str]]:
        """Apply basic code formatting fixes."""
        fixes = []
        
        # Remove trailing whitespace
        lines = content.splitlines()
        cleaned_lines = [line.rstrip() for line in lines]
        
        if lines != cleaned_lines:
            fixes.append("Removed trailing whitespace")
            return '\n'.join(cleaned_lines), fixes
        
        return content, fixes
    
    def _add_type_hints(self, content: str) -> Tuple[str, List[str]]:
        """Add basic type hints where missing."""
        # Simple implementation - in practice would use AST analysis
        return content, []
    
    def _add_docstrings(self, content: str) -> Tuple[str, List[str]]:
        """Add basic docstrings where missing."""
        # Simple implementation - in practice would use AST analysis
        return content, []
    
    def _apply_security_fixes(self, content: str) -> Tuple[str, List[str]]:
        """Apply basic security fixes."""
        fixes = []
        
        # Check for common security issues
        if 'eval(' in content:
            fixes.append("Security warning: eval() usage detected")
        if 'exec(' in content:
            fixes.append("Security warning: exec() usage detected")
            
        return content, fixes
    
    def _calculate_quality_score(self, file_path: Path, content: str) -> float:
        """Calculate comprehensive quality score."""
        score = 1.0
        
        # Basic metrics
        lines = content.splitlines()
        if len(lines) > 1000:  # Large file penalty
            score -= 0.1
        
        # Check for docstrings
        if '"""' not in content and "'''" not in content:
            score -= 0.2
        
        # Check for type hints
        if '->' not in content and ': ' not in content:
            score -= 0.1
        
        # Check for error handling
        if 'try:' not in content:
            score -= 0.15
        
        return max(0.0, score)


class ProgressiveQualityGates:
    """Progressive quality gates that evolve and adapt over time."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.validator = AutomatedQualityValidator(config)
        self.quality_history = []
        self.gate_definitions = {}
        self.evolution_data = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize core quality gates
        self._initialize_gates()
        
        logger.info("Progressive Quality Gates initialized")
    
    def _initialize_gates(self):
        """Initialize core quality gates."""
        self.gate_definitions = {
            'code_structure': {
                'weight': 0.2,
                'threshold': 0.8,
                'validator': self._validate_code_structure,
                'auto_fix': True,
                'evolution_enabled': True
            },
            'testing': {
                'weight': 0.25,
                'threshold': 0.85,
                'validator': self._validate_testing,
                'auto_fix': False,
                'evolution_enabled': True
            },
            'security': {
                'weight': 0.2,
                'threshold': 0.95,
                'validator': self._validate_security,
                'auto_fix': True,
                'evolution_enabled': False  # Security gates don't evolve
            },
            'performance': {
                'weight': 0.15,
                'threshold': 0.8,
                'validator': self._validate_performance,
                'auto_fix': False,
                'evolution_enabled': True
            },
            'documentation': {
                'weight': 0.1,
                'threshold': 0.75,
                'validator': self._validate_documentation,
                'auto_fix': True,
                'evolution_enabled': True
            },
            'maintainability': {
                'weight': 0.1,
                'threshold': 0.8,
                'validator': self._validate_maintainability,
                'auto_fix': True,
                'evolution_enabled': True
            }
        }
    
    async def run_progressive_gates(self, target_path: Path) -> List[QualityGateResult]:
        """Run all quality gates progressively with evolution."""
        logger.info(f"Running progressive quality gates on {target_path}")
        
        # Collect all Python files
        python_files = []
        if target_path.is_file():
            python_files = [target_path]
        else:
            python_files = list(target_path.rglob("*.py"))
        
        if not python_files:
            logger.warning(f"No Python files found in {target_path}")
            return []
        
        # Run gates in parallel
        tasks = []
        for gate_name, gate_config in self.gate_definitions.items():
            task = asyncio.create_task(
                self._run_gate(gate_name, gate_config, python_files)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and flatten results
        quality_results = []
        for result in results:
            if isinstance(result, list):
                quality_results.extend(result)
            elif isinstance(result, QualityGateResult):
                quality_results.append(result)
        
        # Store results for evolution
        self.quality_history.append({
            'timestamp': datetime.now(),
            'results': quality_results,
            'overall_score': self._calculate_overall_score(quality_results)
        })
        
        # Trigger evolution if needed
        await self._evolve_gates(quality_results)
        
        return quality_results
    
    async def _run_gate(self, gate_name: str, gate_config: Dict, files: List[Path]) -> List[QualityGateResult]:
        """Run a specific quality gate."""
        logger.info(f"Running {gate_name} quality gate")
        
        gate_results = []
        validator = gate_config['validator']
        
        # Run validation on all files
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                loop.run_in_executor(executor, validator, file_path)
                for file_path in files
            ]
            
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            for result in results:
                if isinstance(result, QualityGateResult):
                    gate_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Gate {gate_name} failed: {result}")
        
        return gate_results
    
    def _validate_code_structure(self, file_path: Path) -> QualityGateResult:
        """Validate code structure and apply automated fixes."""
        return self.validator.validate_and_fix(file_path)
    
    def _validate_testing(self, file_path: Path) -> QualityGateResult:
        """Validate testing requirements."""
        start_time = time.time()
        
        try:
            content = file_path.read_text()
            score = 0.5  # Base score
            suggestions = []
            
            # Check if this is a test file
            if 'test_' in file_path.name or file_path.parent.name == 'tests':
                score += 0.3
                
                # Check for test functions
                if 'def test_' in content:
                    score += 0.2
                else:
                    suggestions.append("Add test functions starting with 'test_'")
            else:
                # Check if corresponding test file exists
                test_file_patterns = [
                    file_path.parent / f"test_{file_path.name}",
                    file_path.parent / "tests" / f"test_{file_path.name}",
                    Path("tests") / f"test_{file_path.name}"
                ]
                
                test_exists = any(p.exists() for p in test_file_patterns)
                if test_exists:
                    score += 0.3
                else:
                    suggestions.append("Create corresponding test file")
            
            status = QualityGateStatus.PASSED if score >= 0.8 else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="testing",
                status=status,
                score=score,
                execution_time=time.time() - start_time,
                details={"file_type": "test" if "test_" in file_path.name else "source"},
                suggestions=suggestions,
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Testing validation failed for {file_path}: {e}")
            return QualityGateResult(
                gate_name="testing",
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"error": str(e)},
                suggestions=["Manual review required"],
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
    
    def _validate_security(self, file_path: Path) -> QualityGateResult:
        """Validate security requirements."""
        start_time = time.time()
        
        try:
            content = file_path.read_text()
            score = 1.0
            security_issues = []
            auto_fixes = []
            
            # Security pattern checks
            dangerous_patterns = {
                'eval(': 'Use of eval() is dangerous',
                'exec(': 'Use of exec() is dangerous', 
                'subprocess.call(': 'Use subprocess.run() instead of call()',
                'os.system(': 'Use subprocess instead of os.system()',
                'pickle.load(': 'Pickle can execute arbitrary code',
                'yaml.load(': 'Use yaml.safe_load() instead',
            }
            
            for pattern, message in dangerous_patterns.items():
                if pattern in content:
                    score -= 0.2
                    security_issues.append(message)
            
            # Check for hardcoded secrets
            import re
            secret_patterns = [
                r'password\s*=\s*["\'][\w]+["\']',
                r'api_key\s*=\s*["\'][\w]+["\']',
                r'secret\s*=\s*["\'][\w]+["\']'
            ]
            
            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    score -= 0.3
                    security_issues.append("Potential hardcoded secret detected")
            
            status = QualityGateStatus.PASSED if score >= 0.9 else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="security",
                status=status,
                score=max(0.0, score),
                execution_time=time.time() - start_time,
                details={"issues_found": len(security_issues)},
                suggestions=security_issues,
                auto_fixes_applied=auto_fixes,
                evolution_data={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Security validation failed for {file_path}: {e}")
            return QualityGateResult(
                gate_name="security",
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"error": str(e)},
                suggestions=["Manual security review required"],
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
    
    def _validate_performance(self, file_path: Path) -> QualityGateResult:
        """Validate performance requirements."""
        start_time = time.time()
        
        try:
            content = file_path.read_text()
            score = 1.0
            suggestions = []
            
            # Basic performance checks
            lines = content.splitlines()
            if len(lines) > 500:  # Large file
                score -= 0.1
                suggestions.append("Consider breaking large file into smaller modules")
            
            # Check for common performance anti-patterns
            if 'for i in range(len(' in content:
                score -= 0.1
                suggestions.append("Use 'for item in list' instead of 'for i in range(len(list))'")
            
            if '.append(' in content and 'for ' in content:
                suggestions.append("Consider using list comprehension instead of append in loop")
            
            status = QualityGateStatus.PASSED if score >= 0.8 else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="performance",
                status=status,
                score=score,
                execution_time=time.time() - start_time,
                details={"line_count": len(lines)},
                suggestions=suggestions,
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Performance validation failed for {file_path}: {e}")
            return QualityGateResult(
                gate_name="performance",
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"error": str(e)},
                suggestions=["Manual performance review required"],
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
    
    def _validate_documentation(self, file_path: Path) -> QualityGateResult:
        """Validate documentation requirements."""
        start_time = time.time()
        
        try:
            content = file_path.read_text()
            score = 0.0
            auto_fixes = []
            suggestions = []
            
            # Check for module docstring
            if content.strip().startswith('"""') or content.strip().startswith("'''"):
                score += 0.3
            else:
                suggestions.append("Add module docstring at the top of the file")
            
            # Check for function docstrings
            import re
            functions = re.findall(r'def\s+\w+\(', content)
            if functions:
                docstring_pattern = r'def\s+\w+\([^)]*\):\s*\n\s*"""[^"]*"""'
                documented_functions = len(re.findall(docstring_pattern, content))
                score += 0.5 * (documented_functions / len(functions))
                
                if documented_functions < len(functions):
                    suggestions.append(f"Add docstrings to {len(functions) - documented_functions} functions")
            else:
                score += 0.5  # No functions to document
            
            # Check for class docstrings
            classes = re.findall(r'class\s+\w+', content)
            if classes:
                class_docstring_pattern = r'class\s+\w+[^:]*:\s*\n\s*"""[^"]*"""'
                documented_classes = len(re.findall(class_docstring_pattern, content))
                score += 0.2 * (documented_classes / len(classes))
                
                if documented_classes < len(classes):
                    suggestions.append(f"Add docstrings to {len(classes) - documented_classes} classes")
            else:
                score += 0.2  # No classes to document
            
            status = QualityGateStatus.PASSED if score >= 0.75 else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="documentation",
                status=status,
                score=score,
                execution_time=time.time() - start_time,
                details={
                    "functions": len(functions) if 'functions' in locals() else 0,
                    "classes": len(classes) if 'classes' in locals() else 0
                },
                suggestions=suggestions,
                auto_fixes_applied=auto_fixes,
                evolution_data={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Documentation validation failed for {file_path}: {e}")
            return QualityGateResult(
                gate_name="documentation",
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"error": str(e)},
                suggestions=["Manual documentation review required"],
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
    
    def _validate_maintainability(self, file_path: Path) -> QualityGateResult:
        """Validate maintainability requirements."""
        start_time = time.time()
        
        try:
            content = file_path.read_text()
            score = 1.0
            suggestions = []
            
            lines = content.splitlines()
            
            # Complexity checks
            if len(lines) > 300:
                score -= 0.2
                suggestions.append("File is getting large, consider refactoring")
            
            # Check for deeply nested code
            max_indent = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    max_indent = max(max_indent, indent // 4)  # Assuming 4-space indents
            
            if max_indent > 4:
                score -= 0.3
                suggestions.append("Reduce nesting depth - consider extracting functions")
            
            # Check for long functions
            import re
            functions = re.findall(r'def\s+\w+\([^)]*\):(.*?)(?=def|\nclass|\Z)', content, re.DOTALL)
            for func_body in functions:
                func_lines = func_body.count('\n')
                if func_lines > 50:
                    score -= 0.2
                    suggestions.append("Some functions are very long, consider breaking them down")
                    break
            
            status = QualityGateStatus.PASSED if score >= 0.8 else QualityGateStatus.FAILED
            
            return QualityGateResult(
                gate_name="maintainability",
                status=status,
                score=max(0.0, score),
                execution_time=time.time() - start_time,
                details={
                    "line_count": len(lines),
                    "max_nesting": max_indent
                },
                suggestions=suggestions,
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Maintainability validation failed for {file_path}: {e}")
            return QualityGateResult(
                gate_name="maintainability",
                status=QualityGateStatus.FAILED,
                score=0.0,
                execution_time=time.time() - start_time,
                details={"error": str(e)},
                suggestions=["Manual maintainability review required"],
                auto_fixes_applied=[],
                evolution_data={},
                timestamp=datetime.now()
            )
    
    def _calculate_overall_score(self, results: List[QualityGateResult]) -> float:
        """Calculate overall quality score from gate results."""
        if not results:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            gate_config = self.gate_definitions.get(result.gate_name, {})
            weight = gate_config.get('weight', 0.1)
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _evolve_gates(self, results: List[QualityGateResult]):
        """Evolve quality gates based on results and patterns."""
        logger.info("Analyzing quality gate evolution opportunities")
        
        # Analyze patterns in failed gates
        failure_patterns = {}
        for result in results:
            if result.status == QualityGateStatus.FAILED:
                gate_name = result.gate_name
                if gate_name not in failure_patterns:
                    failure_patterns[gate_name] = []
                failure_patterns[gate_name].append(result)
        
        # Evolve gates that are consistently failing or passing
        for gate_name, gate_config in self.gate_definitions.items():
            if not gate_config.get('evolution_enabled', True):
                continue
                
            gate_results = [r for r in results if r.gate_name == gate_name]
            if len(gate_results) < 5:  # Need minimum data for evolution
                continue
            
            # Calculate success rate
            success_rate = len([r for r in gate_results if r.status == QualityGateStatus.PASSED]) / len(gate_results)
            
            # Evolve threshold based on success patterns
            if success_rate > 0.95:  # Too easy
                old_threshold = gate_config['threshold']
                gate_config['threshold'] = min(0.99, old_threshold + 0.05)
                logger.info(f"Evolved {gate_name} threshold from {old_threshold} to {gate_config['threshold']}")
            elif success_rate < 0.5:  # Too hard
                old_threshold = gate_config['threshold']  
                gate_config['threshold'] = max(0.5, old_threshold - 0.05)
                logger.info(f"Evolved {gate_name} threshold from {old_threshold} to {gate_config['threshold']}")
        
        # Store evolution data
        self.evolution_data[datetime.now().isoformat()] = {
            'failure_patterns': failure_patterns,
            'gate_configs': self.gate_definitions.copy(),
            'overall_health': self._calculate_overall_score(results)
        }


class QualityEvolutionManager:
    """Manages long-term evolution of quality gates across the codebase."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.progressive_gates = ProgressiveQualityGates()
        self.evolution_history = []
        self.performance_trends = {}
        
    async def continuous_evolution_cycle(self, interval_hours: int = 24):
        """Run continuous evolution cycles."""
        logger.info(f"Starting continuous evolution cycle every {interval_hours} hours")
        
        while True:
            try:
                # Run comprehensive quality analysis
                results = await self.progressive_gates.run_progressive_gates(self.base_path)
                
                # Analyze trends
                self._analyze_quality_trends(results)
                
                # Generate evolution report
                report = self._generate_evolution_report(results)
                
                # Save report
                report_path = self.base_path / f"quality_evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                report_path.write_text(json.dumps(report, indent=2, default=str))
                
                logger.info(f"Evolution cycle completed. Report saved to {report_path}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Evolution cycle failed: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    def _analyze_quality_trends(self, results: List[QualityGateResult]):
        """Analyze quality trends over time."""
        current_time = datetime.now()
        
        # Group results by gate type
        gate_scores = {}
        for result in results:
            if result.gate_name not in gate_scores:
                gate_scores[result.gate_name] = []
            gate_scores[result.gate_name].append(result.score)
        
        # Calculate trends
        for gate_name, scores in gate_scores.items():
            if gate_name not in self.performance_trends:
                self.performance_trends[gate_name] = []
            
            avg_score = sum(scores) / len(scores)
            self.performance_trends[gate_name].append({
                'timestamp': current_time,
                'average_score': avg_score,
                'sample_count': len(scores),
                'min_score': min(scores),
                'max_score': max(scores)
            })
    
    def _generate_evolution_report(self, results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate comprehensive evolution report."""
        return {
            'timestamp': datetime.now(),
            'overall_health': self.progressive_gates._calculate_overall_score(results),
            'gate_results': [asdict(result) for result in results],
            'evolution_recommendations': self._generate_recommendations(results),
            'trend_analysis': self.performance_trends,
            'next_evolution_targets': self._identify_evolution_targets(results)
        }
    
    def _generate_recommendations(self, results: List[QualityGateResult]) -> List[str]:
        """Generate evolution recommendations."""
        recommendations = []
        
        # Identify consistently failing gates
        gate_failures = {}
        for result in results:
            if result.status == QualityGateStatus.FAILED:
                if result.gate_name not in gate_failures:
                    gate_failures[result.gate_name] = 0
                gate_failures[result.gate_name] += 1
        
        for gate_name, failure_count in gate_failures.items():
            if failure_count > len(results) * 0.3:  # More than 30% failure rate
                recommendations.append(f"Focus on improving {gate_name} - high failure rate ({failure_count} failures)")
        
        return recommendations
    
    def _identify_evolution_targets(self, results: List[QualityGateResult]) -> List[str]:
        """Identify targets for next evolution cycle."""
        targets = []
        
        # Find gates with improvement potential
        gate_scores = {}
        for result in results:
            if result.gate_name not in gate_scores:
                gate_scores[result.gate_name] = []
            gate_scores[result.gate_name].append(result.score)
        
        for gate_name, scores in gate_scores.items():
            avg_score = sum(scores) / len(scores)
            if 0.6 < avg_score < 0.8:  # Room for improvement
                targets.append(f"{gate_name} (current avg: {avg_score:.2f})")
        
        return targets