#!/usr/bin/env python3
"""
Log Analysis and Aggregation Tool
Analyzes logs from the video synthesis pipeline to provide insights,
performance metrics, error patterns, and debugging information.
"""

import os
import json
import glob
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import statistics
import argparse


class LogAnalyzer:
    """Analyzes pipeline logs to provide insights and debugging information."""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.parsed_logs = []
        self.analysis_results = {}
        
    def load_logs(self, date_filter: Optional[str] = None, 
                  component_filter: Optional[str] = None,
                  level_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load and parse log files with optional filters."""
        logs = []
        
        # Get all log files
        log_files = []
        for log_file in self.log_directory.rglob("*.log"):
            # Apply date filter if specified
            if date_filter and date_filter not in log_file.name:
                continue
            
            # Apply component filter if specified
            if component_filter and component_filter not in log_file.name:
                continue
                
            log_files.append(log_file)
        
        # Parse each log file
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            log_entry = json.loads(line.strip())
                            
                            # Apply level filter if specified
                            if level_filter and log_entry.get('level') != level_filter:
                                continue
                            
                            # Add file information
                            log_entry['source_file'] = str(log_file)
                            log_entry['line_number'] = line_num
                            
                            logs.append(log_entry)
                            
                        except json.JSONDecodeError:
                            # Skip non-JSON lines
                            continue
                            
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")
                continue
        
        # Sort logs by timestamp
        logs.sort(key=lambda x: x.get('timestamp', ''))
        
        self.parsed_logs = logs
        return logs
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics from logs."""
        performance_data = {
            'api_endpoints': defaultdict(list),
            'ai_generators': defaultdict(list),
            'pipeline_stages': defaultdict(list),
            'execution_times': [],
            'memory_usage': [],
            'slow_operations': []
        }
        
        for log in self.parsed_logs:
            execution_time = log.get('execution_time_ms')
            memory_usage = log.get('memory_usage_mb')
            component = log.get('component', 'unknown')
            event = log.get('event', 'unknown')
            
            # Collect execution times
            if execution_time is not None:
                performance_data['execution_times'].append({
                    'component': component,
                    'event': event,
                    'execution_time': execution_time,
                    'timestamp': log.get('timestamp')
                })
                
                # Categorize by component type
                if component == 'api_server':
                    endpoint = log.get('metadata', {}).get('endpoint', 'unknown')
                    performance_data['api_endpoints'][endpoint].append(execution_time)
                elif 'generator' in component:
                    performance_data['ai_generators'][component].append(execution_time)
                elif component == 'pipeline_stage':
                    stage = log.get('metadata', {}).get('stage_name', 'unknown')
                    performance_data['pipeline_stages'][stage].append(execution_time)
                
                # Flag slow operations (> 5 seconds)
                if execution_time > 5000:
                    performance_data['slow_operations'].append({
                        'component': component,
                        'event': event,
                        'execution_time': execution_time,
                        'timestamp': log.get('timestamp'),
                        'session_id': log.get('session_id'),
                        'metadata': log.get('metadata', {})
                    })
            
            # Collect memory usage
            if memory_usage is not None:
                performance_data['memory_usage'].append({
                    'component': component,
                    'memory_usage': memory_usage,
                    'timestamp': log.get('timestamp')
                })
        
        # Calculate statistics
        stats = {}
        
        # API endpoint statistics
        for endpoint, times in performance_data['api_endpoints'].items():
            if times:
                stats[f'api_{endpoint}'] = {
                    'count': len(times),
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'median_time': statistics.median(times)
                }
        
        # AI generator statistics
        for generator, times in performance_data['ai_generators'].items():
            if times:
                stats[f'ai_{generator}'] = {
                    'count': len(times),
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'median_time': statistics.median(times)
                }
        
        # Pipeline stage statistics
        for stage, times in performance_data['pipeline_stages'].items():
            if times:
                stats[f'stage_{stage}'] = {
                    'count': len(times),
                    'avg_time': statistics.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'median_time': statistics.median(times)
                }
        
        self.analysis_results['performance'] = {
            'raw_data': performance_data,
            'statistics': stats,
            'slow_operations_count': len(performance_data['slow_operations']),
            'total_operations': len(performance_data['execution_times'])
        }
        
        return self.analysis_results['performance']
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns and frequencies."""
        error_data = {
            'error_counts': defaultdict(int),
            'error_patterns': defaultdict(list),
            'error_timeline': [],
            'critical_errors': [],
            'component_errors': defaultdict(int),
            'session_errors': defaultdict(list)
        }
        
        for log in self.parsed_logs:
            level = log.get('level', '').upper()
            
            if level in ['ERROR', 'CRITICAL']:
                component = log.get('component', 'unknown')
                event = log.get('event', 'unknown')
                message = log.get('message', '')
                session_id = log.get('session_id')
                timestamp = log.get('timestamp')
                error_details = log.get('error_details', {})
                
                # Count errors by type
                error_key = f"{component}_{event}"
                error_data['error_counts'][error_key] += 1
                
                # Track error patterns
                error_data['error_patterns'][error_key].append({
                    'timestamp': timestamp,
                    'session_id': session_id,
                    'message': message,
                    'error_details': error_details,
                    'metadata': log.get('metadata', {})
                })
                
                # Timeline of errors
                error_data['error_timeline'].append({
                    'timestamp': timestamp,
                    'level': level,
                    'component': component,
                    'event': event,
                    'message': message,
                    'session_id': session_id
                })
                
                # Critical errors
                if level == 'CRITICAL':
                    error_data['critical_errors'].append({
                        'timestamp': timestamp,
                        'component': component,
                        'event': event,
                        'message': message,
                        'session_id': session_id,
                        'error_details': error_details
                    })
                
                # Component error counts
                error_data['component_errors'][component] += 1
                
                # Session error tracking
                if session_id:
                    error_data['session_errors'][session_id].append({
                        'timestamp': timestamp,
                        'component': component,
                        'event': event,
                        'message': message,
                        'level': level
                    })
        
        # Sort timeline by timestamp
        error_data['error_timeline'].sort(key=lambda x: x['timestamp'])
        
        # Find most problematic sessions
        problematic_sessions = sorted(
            error_data['session_errors'].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]
        
        self.analysis_results['errors'] = {
            'total_errors': sum(error_data['error_counts'].values()),
            'critical_errors': len(error_data['critical_errors']),
            'error_types': len(error_data['error_counts']),
            'most_common_errors': Counter(error_data['error_counts']).most_common(10),
            'component_error_distribution': dict(error_data['component_errors']),
            'problematic_sessions': problematic_sessions[:5],
            'error_patterns': dict(error_data['error_patterns']),
            'recent_errors': error_data['error_timeline'][-20:] if error_data['error_timeline'] else []
        }
        
        return self.analysis_results['errors']
    
    def analyze_sessions(self) -> Dict[str, Any]:
        """Analyze session patterns and user flows."""
        session_data = defaultdict(lambda: {
            'events': [],
            'start_time': None,
            'end_time': None,
            'duration': 0,
            'error_count': 0,
            'success_count': 0,
            'components_used': set(),
            'user_id': None,
            'script_generations': 0,
            'thumbnail_generations': 0,
            'file_uploads': 0,
            'video_generations': 0
        })
        
        for log in self.parsed_logs:
            session_id = log.get('session_id')
            if not session_id:
                continue
            
            session = session_data[session_id]
            timestamp = log.get('timestamp')
            component = log.get('component', 'unknown')
            event = log.get('event', 'unknown')
            level = log.get('level', 'INFO')
            
            # Track events
            session['events'].append({
                'timestamp': timestamp,
                'component': component,
                'event': event,
                'level': level,
                'message': log.get('message', ''),
                'execution_time': log.get('execution_time_ms')
            })
            
            # Track timing
            if session['start_time'] is None or timestamp < session['start_time']:
                session['start_time'] = timestamp
            if session['end_time'] is None or timestamp > session['end_time']:
                session['end_time'] = timestamp
            
            # Count errors and successes
            if level in ['ERROR', 'CRITICAL']:
                session['error_count'] += 1
            elif level == 'INFO' and 'success' in event.lower():
                session['success_count'] += 1
            
            # Track components used
            session['components_used'].add(component)
            
            # Track user ID
            if log.get('user_id') and session['user_id'] is None:
                session['user_id'] = log.get('user_id')
            
            # Track specific activities
            if event == 'script_generation_complete':
                session['script_generations'] += 1
            elif event == 'thumbnail_generation_complete':
                session['thumbnail_generations'] += 1
            elif event == 'file_upload_success':
                session['file_uploads'] += 1
            elif event == 'video_generation_complete':
                session['video_generations'] += 1
        
        # Calculate durations and success rates
        session_stats = []
        for session_id, data in session_data.items():
            if data['start_time'] and data['end_time']:
                start = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
                data['duration'] = (end - start).total_seconds()
            
            data['components_used'] = list(data['components_used'])
            data['success_rate'] = data['success_count'] / max(1, data['success_count'] + data['error_count'])
            
            session_stats.append({
                'session_id': session_id,
                'duration': data['duration'],
                'error_count': data['error_count'],
                'success_count': data['success_count'],
                'success_rate': data['success_rate'],
                'components_used': len(data['components_used']),
                'total_events': len(data['events']),
                'script_generations': data['script_generations'],
                'thumbnail_generations': data['thumbnail_generations'],
                'file_uploads': data['file_uploads'],
                'video_generations': data['video_generations']
            })
        
        # Sort by various metrics
        longest_sessions = sorted(session_stats, key=lambda x: x['duration'], reverse=True)[:10]
        most_active_sessions = sorted(session_stats, key=lambda x: x['total_events'], reverse=True)[:10]
        most_errors = sorted(session_stats, key=lambda x: x['error_count'], reverse=True)[:10]
        
        self.analysis_results['sessions'] = {
            'total_sessions': len(session_data),
            'avg_duration': statistics.mean([s['duration'] for s in session_stats if s['duration'] > 0]),
            'avg_events_per_session': statistics.mean([s['total_events'] for s in session_stats]),
            'avg_success_rate': statistics.mean([s['success_rate'] for s in session_stats]),
            'longest_sessions': longest_sessions,
            'most_active_sessions': most_active_sessions,
            'most_problematic_sessions': most_errors,
            'session_details': dict(session_data)
        }
        
        return self.analysis_results['sessions']
    
    def analyze_api_usage(self) -> Dict[str, Any]:
        """Analyze API usage patterns and costs."""
        api_data = {
            'gemini_calls': [],
            'dalle_calls': [],
            'stability_calls': [],
            'endpoint_usage': defaultdict(int),
            'success_rates': defaultdict(lambda: {'success': 0, 'failure': 0}),
            'cost_estimation': {
                'gemini': 0,
                'dalle': 0,
                'stability': 0
            }
        }
        
        # Cost estimates (example rates)
        costs = {
            'gemini': 0.001,  # per 1000 tokens
            'dalle': 0.020,   # per image
            'stability': 0.010  # per image
        }
        
        for log in self.parsed_logs:
            component = log.get('component', '')
            event = log.get('event', '')
            metadata = log.get('metadata', {})
            
            # API endpoint usage
            if component == 'api_server' and event == 'api_request_complete':
                endpoint = metadata.get('endpoint', 'unknown')
                api_data['endpoint_usage'][endpoint] += 1
                
                status = metadata.get('status_code', 500)
                if 200 <= status < 300:
                    api_data['success_rates'][endpoint]['success'] += 1
                else:
                    api_data['success_rates'][endpoint]['failure'] += 1
            
            # AI API calls
            elif component == 'gemini_generator':
                if event == 'gemini_api_call_success':
                    api_data['gemini_calls'].append({
                        'timestamp': log.get('timestamp'),
                        'execution_time': log.get('execution_time_ms'),
                        'success': True,
                        'topic': metadata.get('topic'),
                        'word_count': metadata.get('word_count', 0),
                        'generated_length': metadata.get('generated_length', 0)
                    })
                    # Estimate cost based on output length
                    tokens = metadata.get('word_count', 0) * 1.3  # Rough token estimate
                    api_data['cost_estimation']['gemini'] += (tokens / 1000) * costs['gemini']
                    
                elif event == 'gemini_api_call_failed':
                    api_data['gemini_calls'].append({
                        'timestamp': log.get('timestamp'),
                        'execution_time': log.get('execution_time_ms'),
                        'success': False,
                        'error_type': metadata.get('error_type')
                    })
            
            elif component == 'thumbnail_generator':
                if event == 'dalle_thumbnail_generated':
                    api_data['dalle_calls'].append({
                        'timestamp': log.get('timestamp'),
                        'execution_time': log.get('execution_time_ms'),
                        'success': True,
                        'filename': metadata.get('filename'),
                        'file_size': metadata.get('file_size', 0)
                    })
                    api_data['cost_estimation']['dalle'] += costs['dalle']
                    
                elif event == 'dalle_api_call_failed':
                    api_data['dalle_calls'].append({
                        'timestamp': log.get('timestamp'),
                        'execution_time': log.get('execution_time_ms'),
                        'success': False,
                        'error_type': metadata.get('error_type')
                    })
        
        # Calculate success rates
        success_rates = {}
        for endpoint, stats in api_data['success_rates'].items():
            total = stats['success'] + stats['failure']
            if total > 0:
                success_rates[endpoint] = stats['success'] / total
        
        self.analysis_results['api_usage'] = {
            'endpoint_usage': dict(api_data['endpoint_usage']),
            'success_rates': success_rates,
            'gemini_calls': len(api_data['gemini_calls']),
            'dalle_calls': len(api_data['dalle_calls']),
            'total_estimated_cost': sum(api_data['cost_estimation'].values()),
            'cost_breakdown': api_data['cost_estimation'],
            'gemini_success_rate': len([c for c in api_data['gemini_calls'] if c['success']]) / max(1, len(api_data['gemini_calls'])),
            'dalle_success_rate': len([c for c in api_data['dalle_calls'] if c['success']]) / max(1, len(api_data['dalle_calls']))
        }
        
        return self.analysis_results['api_usage']
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("VIDEO SYNTHESIS PIPELINE - LOG ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total log entries analyzed: {len(self.parsed_logs)}")
        report.append("")
        
        # Performance Analysis
        if 'performance' in self.analysis_results:
            perf = self.analysis_results['performance']
            report.append("PERFORMANCE ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total operations tracked: {perf['total_operations']}")
            report.append(f"Slow operations (>5s): {perf['slow_operations_count']}")
            report.append("")
            
            report.append("Top Performance Statistics:")
            for key, stats in perf['statistics'].items():
                report.append(f"  {key}:")
                report.append(f"    Count: {stats['count']}")
                report.append(f"    Average: {stats['avg_time']:.2f}ms")
                report.append(f"    Min/Max: {stats['min_time']:.2f}ms / {stats['max_time']:.2f}ms")
                report.append("")
        
        # Error Analysis
        if 'errors' in self.analysis_results:
            errors = self.analysis_results['errors']
            report.append("ERROR ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total errors: {errors['total_errors']}")
            report.append(f"Critical errors: {errors['critical_errors']}")
            report.append(f"Error types: {errors['error_types']}")
            report.append("")
            
            report.append("Most Common Errors:")
            for error_type, count in errors['most_common_errors']:
                report.append(f"  {error_type}: {count} occurrences")
            report.append("")
            
            report.append("Component Error Distribution:")
            for component, count in errors['component_error_distribution'].items():
                report.append(f"  {component}: {count} errors")
            report.append("")
        
        # Session Analysis
        if 'sessions' in self.analysis_results:
            sessions = self.analysis_results['sessions']
            report.append("SESSION ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total sessions: {sessions['total_sessions']}")
            report.append(f"Average duration: {sessions['avg_duration']:.2f} seconds")
            report.append(f"Average events per session: {sessions['avg_events_per_session']:.2f}")
            report.append(f"Average success rate: {sessions['avg_success_rate']:.2%}")
            report.append("")
            
            report.append("Longest Sessions:")
            for session in sessions['longest_sessions'][:5]:
                report.append(f"  {session['session_id']}: {session['duration']:.2f}s, {session['total_events']} events")
            report.append("")
        
        # API Usage Analysis
        if 'api_usage' in self.analysis_results:
            api = self.analysis_results['api_usage']
            report.append("API USAGE ANALYSIS")
            report.append("-" * 50)
            report.append(f"Total estimated cost: ${api['total_estimated_cost']:.4f}")
            report.append(f"Gemini calls: {api['gemini_calls']} (Success: {api['gemini_success_rate']:.2%})")
            report.append(f"DALL-E calls: {api['dalle_calls']} (Success: {api['dalle_success_rate']:.2%})")
            report.append("")
            
            report.append("Cost Breakdown:")
            for service, cost in api['cost_breakdown'].items():
                report.append(f"  {service}: ${cost:.4f}")
            report.append("")
            
            report.append("Endpoint Usage:")
            for endpoint, count in api['endpoint_usage'].items():
                success_rate = api['success_rates'].get(endpoint, 0)
                report.append(f"  {endpoint}: {count} calls (Success: {success_rate:.2%})")
            report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append("End of Report")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text
    
    def run_full_analysis(self, date_filter: str = None, 
                         component_filter: str = None,
                         level_filter: str = None) -> Dict[str, Any]:
        """Run complete analysis with all modules."""
        print("Loading logs...")
        self.load_logs(date_filter, component_filter, level_filter)
        
        print("Analyzing performance...")
        self.analyze_performance()
        
        print("Analyzing errors...")
        self.analyze_errors()
        
        print("Analyzing sessions...")
        self.analyze_sessions()
        
        print("Analyzing API usage...")
        self.analyze_api_usage()
        
        return self.analysis_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze video synthesis pipeline logs')
    parser.add_argument('--log-dir', default='logs', help='Directory containing log files')
    parser.add_argument('--date', help='Filter logs by date (YYYY-MM-DD)')
    parser.add_argument('--component', help='Filter logs by component')
    parser.add_argument('--level', help='Filter logs by level')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='Report format')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = LogAnalyzer(args.log_dir)
    
    # Run analysis
    results = analyzer.run_full_analysis(
        date_filter=args.date,
        component_filter=args.component,
        level_filter=args.level
    )
    
    # Generate report
    if args.format == 'json':
        output = json.dumps(results, indent=2, default=str)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
        else:
            print(output)
    else:
        report = analyzer.generate_report(args.output)
        if not args.output:
            print(report)


if __name__ == "__main__":
    main()