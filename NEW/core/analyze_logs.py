#!/usr/bin/env python3
"""
Log Analysis Utility for Video Synthesis Pipeline
Analyzes log files to identify patterns, errors, and performance issues
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict, Counter
import argparse


class LogAnalyzer:
    """Analyzes pipeline log files to identify issues and patterns."""
    
    def __init__(self, logs_dir: str = None):
        """Initialize log analyzer.
        
        Args:
            logs_dir: Directory containing log files. Defaults to NEW/core/logs/
        """
        if logs_dir is None:
            self.logs_dir = Path(__file__).parent / "logs"
        else:
            self.logs_dir = Path(logs_dir)
        
        self.analysis_results = {}
    
    def analyze_all_logs(self) -> Dict[str, Any]:
        """Analyze all log files and return comprehensive results."""
        print("Search Analyzing pipeline logs...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'logs_directory': str(self.logs_dir),
            'sadtalker_analysis': self.analyze_sadtalker_logs(),
            'error_analysis': self.analyze_error_logs(),
            'performance_analysis': self.analyze_performance_logs(),
            'cli_analysis': self.analyze_cli_logs(),
            'summary': {}
        }
        
        # Generate summary
        results['summary'] = self.generate_summary(results)
        
        return results
    
    def analyze_sadtalker_logs(self) -> Dict[str, Any]:
        """Analyze SadTalker-specific debug logs."""
        sadtalker_log_files = list(self.logs_dir.glob("debug/sadtalker_debug_*.log"))
        
        analysis = {
            'files_found': len(sadtalker_log_files),
            'total_processing_attempts': 0,
            'duration_issues': [],
            'successful_generations': [],
            'failed_generations': [],
            'common_errors': Counter(),
            'processing_times': [],
            'video_properties': []
        }
        
        for log_file in sadtalker_log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                # Look for processing attempts
                processing_starts = re.findall(r'STARTING SadTalker lip-sync processing.*?Context: ({.*?})', content, re.DOTALL)
                analysis['total_processing_attempts'] += len(processing_starts)
                
                # Look for duration issues
                duration_issues = re.findall(r'SadTalker 0\.04s duration issue detected.*?Context: ({.*?})', content, re.DOTALL)
                for issue in duration_issues:
                    try:
                        issue_data = json.loads(issue.replace("'", '"'))
                        analysis['duration_issues'].append(issue_data)
                    except:
                        pass
                
                # Look for successful generations  
                successes = re.findall(r'SadTalker duration looks healthy.*?Context: ({.*?})', content, re.DOTALL)
                for success in successes:
                    try:
                        success_data = json.loads(success.replace("'", '"'))
                        analysis['successful_generations'].append(success_data)
                    except:
                        pass
                
                # Look for errors
                errors = re.findall(r'SadTalker.*?ERROR.*?Context: ({.*?})', content, re.DOTALL)
                for error in errors:
                    try:
                        error_data = json.loads(error.replace("'", '"'))
                        analysis['failed_generations'].append(error_data)
                        # Count error types
                        for err in error_data.get('errors', []):
                            analysis['common_errors'][err] += 1
                    except:
                        pass
                        
            except Exception as e:
                print(f"[WARNING] Error reading {log_file}: {e}")
        
        return analysis
    
    def analyze_error_logs(self) -> Dict[str, Any]:
        """Analyze error logs for patterns."""
        error_log_files = list(self.logs_dir.glob("errors/*.log"))
        
        analysis = {
            'files_found': len(error_log_files),
            'total_errors': 0,
            'critical_errors': 0,
            'error_categories': Counter(),
            'recent_errors': [],
            'pipeline_failures': []
        }
        
        for log_file in error_log_files:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    if ' - ERROR - ' in line or ' - CRITICAL - ' in line:
                        analysis['total_errors'] += 1
                        
                        if ' - CRITICAL - ' in line:
                            analysis['critical_errors'] += 1
                        
                        # Categorize errors
                        if 'SadTalker' in line:
                            analysis['error_categories']['SadTalker'] += 1
                        elif 'Enhancement' in line:
                            analysis['error_categories']['Enhancement'] += 1
                        elif 'Voice' in line:
                            analysis['error_categories']['Voice'] += 1
                        elif 'CLI' in line:
                            analysis['error_categories']['CLI'] += 1
                        elif 'Pipeline' in line:
                            analysis['error_categories']['Pipeline'] += 1
                        else:
                            analysis['error_categories']['Other'] += 1
                        
                        # Keep recent errors (last 10)
                        if len(analysis['recent_errors']) < 10:
                            analysis['recent_errors'].append(line.strip())
                            
            except Exception as e:
                print(f"[WARNING] Error reading {log_file}: {e}")
        
        return analysis
    
    def analyze_performance_logs(self) -> Dict[str, Any]:
        """Analyze performance and timing logs."""
        perf_log_files = list(self.logs_dir.glob("performance/*.log"))
        
        analysis = {
            'files_found': len(perf_log_files),
            'stage_timings': defaultdict(list),
            'total_pipeline_runs': 0,
            'average_times': {},
            'slowest_stages': [],
            'fastest_stages': []
        }
        
        for log_file in perf_log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                # Find stage completions with timing
                stage_timings = re.findall(r'Completed stage: (\w+).*?Duration: ([\d.]+)s', content)
                
                for stage, duration in stage_timings:
                    analysis['stage_timings'][stage].append(float(duration))
                
                # Count pipeline runs
                pipeline_starts = content.count('Starting stage: video_generation_pipeline')
                analysis['total_pipeline_runs'] += pipeline_starts
                        
            except Exception as e:
                print(f"[WARNING] Error reading {log_file}: {e}")
        
        # Calculate averages
        for stage, times in analysis['stage_timings'].items():
            if times:
                analysis['average_times'][stage] = sum(times) / len(times)
        
        # Find slowest and fastest stages
        if analysis['average_times']:
            sorted_stages = sorted(analysis['average_times'].items(), key=lambda x: x[1], reverse=True)
            analysis['slowest_stages'] = sorted_stages[:3]
            analysis['fastest_stages'] = sorted_stages[-3:]
        
        return analysis
    
    def analyze_cli_logs(self) -> Dict[str, Any]:
        """Analyze CLI usage logs."""
        cli_log_files = list(self.logs_dir.glob("cli/*.log"))
        
        analysis = {
            'files_found': len(cli_log_files),
            'total_sessions': 0,
            'successful_sessions': 0,
            'failed_sessions': 0,
            'common_parameters': Counter(),
            'user_errors': []
        }
        
        for log_file in cli_log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                # Count sessions
                sessions = content.count('Initializing CLI Video Generator')
                analysis['total_sessions'] += sessions
                
                # Count successes and failures
                successes = content.count('VIDEO GENERATION SUCCESSFUL')
                failures = content.count('Pipeline execution failed')
                analysis['successful_sessions'] += successes
                analysis['failed_sessions'] += failures
                
                # Find parameter usage
                tone_matches = re.findall(r"'tone': '(\w+)'", content)
                emotion_matches = re.findall(r"'emotion': '(\w+)'", content)
                audience_matches = re.findall(r"'audience': '(\w+)'", content)
                
                for tone in tone_matches:
                    analysis['common_parameters'][f'tone_{tone}'] += 1
                for emotion in emotion_matches:
                    analysis['common_parameters'][f'emotion_{emotion}'] += 1
                for audience in audience_matches:
                    analysis['common_parameters'][f'audience_{audience}'] += 1
                
                # Find user errors
                user_errors = re.findall(r'cli_error.*?Exception: (.*?)\\n', content)
                analysis['user_errors'].extend(user_errors[:5])  # Keep first 5
                        
            except Exception as e:
                print(f"[WARNING] Error reading {log_file}: {e}")
        
        return analysis
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of key findings."""
        sadtalker = results['sadtalker_analysis']
        errors = results['error_analysis']
        performance = results['performance_analysis']
        cli = results['cli_analysis']
        
        summary = {
            'critical_issues': [],
            'recommendations': [],
            'health_score': 0,
            'key_metrics': {}
        }
        
        # Check for critical issues
        if len(sadtalker['duration_issues']) > 0:
            summary['critical_issues'].append(f"SadTalker 0.04s duration issue detected in {len(sadtalker['duration_issues'])} cases")
            summary['recommendations'].append("Fix SadTalker video duration truncation - check audio preprocessing and model parameters")
        
        if errors['critical_errors'] > 0:
            summary['critical_issues'].append(f"{errors['critical_errors']} critical errors found")
            summary['recommendations'].append("Review critical error logs and fix blocking issues")
        
        if cli['failed_sessions'] > cli['successful_sessions']:
            summary['critical_issues'].append(f"More failed CLI sessions ({cli['failed_sessions']}) than successful ({cli['successful_sessions']})")
            summary['recommendations'].append("Improve CLI error handling and user input validation")
        
        # Calculate health score (0-100)
        health_score = 100
        health_score -= len(sadtalker['duration_issues']) * 10  # -10 per duration issue
        health_score -= errors['critical_errors'] * 5  # -5 per critical error
        health_score -= max(0, cli['failed_sessions'] - cli['successful_sessions']) * 3  # -3 per excess failure
        
        summary['health_score'] = max(0, health_score)
        
        # Key metrics
        summary['key_metrics'] = {
            'total_sadtalker_attempts': sadtalker['total_processing_attempts'],
            'sadtalker_success_rate': len(sadtalker['successful_generations']) / max(1, sadtalker['total_processing_attempts']) * 100,
            'total_errors': errors['total_errors'],
            'cli_success_rate': cli['successful_sessions'] / max(1, cli['total_sessions']) * 100,
            'average_pipeline_time': performance['average_times'].get('video_generation_pipeline', 0)
        }
        
        # Add recommendations based on findings
        if summary['key_metrics']['sadtalker_success_rate'] < 80:
            summary['recommendations'].append("SadTalker success rate is low - check model availability and parameters")
        
        if summary['key_metrics']['cli_success_rate'] < 70:
            summary['recommendations'].append("CLI success rate is low - improve user experience and error messages")
        
        return summary
    
    def print_report(self, results: Dict[str, Any]):
        """Print a formatted analysis report."""
        print("\n" + "="*80)
        print("Status: PIPELINE LOG ANALYSIS REPORT")
        print("="*80)
        
        summary = results['summary']
        
        # Health Score
        health_score = summary['health_score']
        health_emoji = "🟢" if health_score >= 80 else "🟡" if health_score >= 60 else "[EMOJI]"
        print(f"\n{health_emoji} OVERALL HEALTH SCORE: {health_score}/100")
        
        # Critical Issues
        if summary['critical_issues']:
            print(f"\n[EMOJI] CRITICAL ISSUES:")
            for issue in summary['critical_issues']:
                print(f"   [ERROR] {issue}")
        else:
            print(f"\n[SUCCESS] NO CRITICAL ISSUES DETECTED")
        
        # Key Metrics
        print(f"\nPerformance KEY METRICS:")
        metrics = summary['key_metrics']
        print(f"   • SadTalker Success Rate: {metrics['sadtalker_success_rate']:.1f}%")
        print(f"   • CLI Success Rate: {metrics['cli_success_rate']:.1f}%")
        print(f"   • Total Processing Attempts: {metrics['total_sadtalker_attempts']}")
        print(f"   • Total Errors: {metrics['total_errors']}")
        print(f"   • Average Pipeline Time: {metrics['average_pipeline_time']:.1f}s")
        
        # SadTalker Analysis
        sadtalker = results['sadtalker_analysis']
        print(f"\nStyle: SADTALKER ANALYSIS:")
        print(f"   • Processing Attempts: {sadtalker['total_processing_attempts']}")
        print(f"   • Duration Issues: {len(sadtalker['duration_issues'])}")
        print(f"   • Successful Generations: {len(sadtalker['successful_generations'])}")
        print(f"   • Failed Generations: {len(sadtalker['failed_generations'])}")
        
        if sadtalker['duration_issues']:
            print(f"   [WARNING] Duration Issue Details:")
            for issue in sadtalker['duration_issues'][:3]:  # Show first 3
                duration = issue.get('actual_duration', 'unknown')
                print(f"      - Duration: {duration}s, Frames: {issue.get('video_frames', 'unknown')}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\nINFO: RECOMMENDATIONS:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Error Categories
        errors = results['error_analysis']
        if errors['error_categories']:
            print(f"\n[ERROR] ERROR CATEGORIES:")
            for category, count in errors['error_categories'].most_common(5):
                print(f"   • {category}: {count} errors")
        
        # Performance Analysis
        performance = results['performance_analysis']
        if performance['slowest_stages']:
            print(f"\nDuration: SLOWEST STAGES:")
            for stage, time in performance['slowest_stages']:
                print(f"   • {stage}: {time:.1f}s average")
        
        print("\n" + "="*80)
        print(f"Report generated: {results['timestamp']}")
        print("="*80 + "\n")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze video synthesis pipeline logs')
    parser.add_argument('--logs-dir', help='Directory containing log files')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.logs_dir)
    results = analyzer.analyze_all_logs()
    
    if not args.quiet:
        analyzer.print_report(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"File: Results saved to: {args.output}")
    
    # Return exit code based on health score
    health_score = results['summary']['health_score']
    if health_score < 60:
        return 1  # Critical issues
    elif health_score < 80:
        return 2  # Warning issues
    else:
        return 0  # Healthy


if __name__ == "__main__":
    exit(main())