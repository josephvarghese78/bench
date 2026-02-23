"""
Performance Test Report Generator
Generates a comprehensive PDF report with graphs from benchmark test results
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import os
import glob
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def get_latest_db_file(db_folder="./db"):
    """Get the most recent database file"""
    db_files = glob.glob(f"{db_folder}/*.db")
    if not db_files:
        raise FileNotFoundError(f"No database files found in {db_folder}")
    latest_db = max(db_files, key=os.path.getctime)
    return latest_db


def load_data_from_db(db_path):
    """Load performance data from SQLite database"""
    print(f"Loading data from: {db_path}")
    conn = sqlite3.connect(db_path)

    query = """
    SELECT suite_name, test_name, max_threads, thread_name, iteration,
           start_time, end_time, samples_started, samples_completed,
           active_threads, think_time, response_time, error_status,
           error_percent, response_status_code, response
    FROM performance ORDER BY start_time
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    df['is_error'] = df['error_status'].apply(lambda x: 1 if x == 'F' else 0)

    print(f"Loaded {len(df)} records")
    return df


def create_summary_stats(df):
    """Generate summary statistics"""
    stats = {
        'Total Requests': len(df),
        'Total Suites': df['suite_name'].nunique(),
        'Max Concurrent Users': df['max_threads'].max(),
        'Total Errors': df['is_error'].sum(),
        'Error Rate (%)': (df['is_error'].sum() / len(df) * 100) if len(df) > 0 else 0,
        'Avg Response Time (ms)': df['response_time'].mean(),
        'Min Response Time (ms)': df['response_time'].min(),
        'Max Response Time (ms)': df['response_time'].max(),
        'Median Response Time (ms)': df['response_time'].median(),
        '95th Percentile (ms)': df['response_time'].quantile(0.95),
        '99th Percentile (ms)': df['response_time'].quantile(0.99),
        'Avg Think Time (ms)': df['think_time'].mean(),
    }
    return stats


def plot_response_time_over_time(df, ax):
    ax.plot(df['start_time'], df['response_time'], alpha=0.6, linewidth=1, color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title('Response Time Over Time', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    mean_rt = df['response_time'].mean()
    ax.axhline(y=mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}ms')
    ax.legend()


def plot_response_time_distribution(df, ax):
    ax.hist(df['response_time'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Response Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    mean_rt = df['response_time'].mean()
    median_rt = df['response_time'].median()
    p95 = df['response_time'].quantile(0.95)
    ax.axvline(x=mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}ms')
    ax.axvline(x=median_rt, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rt:.2f}ms')
    ax.axvline(x=p95, color='orange', linestyle='--', linewidth=2, label=f'95th %ile: {p95:.2f}ms')
    ax.legend()


def plot_response_time_by_users(df, ax):
    grouped = df.groupby('max_threads').agg({'response_time': ['mean', 'median', 'min', 'max']}).reset_index()
    grouped.columns = ['max_threads', 'mean_rt', 'median_rt', 'min_rt', 'max_rt']
    ax.plot(grouped['max_threads'], grouped['mean_rt'], marker='o', linewidth=2, markersize=8, label='Mean', color='blue')
    ax.plot(grouped['max_threads'], grouped['median_rt'], marker='s', linewidth=2, markersize=8, label='Median', color='green')
    ax.fill_between(grouped['max_threads'], grouped['min_rt'], grouped['max_rt'], alpha=0.2, color='gray', label='Min-Max Range')
    ax.set_xlabel('Number of Concurrent Users')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title('Response Time vs Concurrent Users', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_error_rate_over_time(df, ax):
    df_sorted = df.sort_values('start_time')
    window_size = max(1, len(df) // 50)
    df_sorted['window'] = df_sorted.index // window_size
    error_rate = df_sorted.groupby('window').agg({'is_error': 'mean', 'start_time': 'first'}).reset_index()
    error_rate['error_rate_pct'] = error_rate['is_error'] * 100
    ax.plot(error_rate['start_time'], error_rate['error_rate_pct'], linewidth=2, color='red', marker='o', markersize=4)
    ax.fill_between(error_rate['start_time'], 0, error_rate['error_rate_pct'], alpha=0.3, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate Over Time', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def plot_throughput_over_time(df, ax):
    df_sorted = df.sort_values('start_time')
    df_sorted['time_minute'] = df_sorted['start_time'].dt.floor('1min')
    throughput = df_sorted.groupby('time_minute').size().reset_index(name='requests_per_min')
    ax.bar(throughput['time_minute'], throughput['requests_per_min'], width=0.0005, color='green', alpha=0.7, edgecolor='darkgreen')
    ax.set_xlabel('Time')
    ax.set_ylabel('Requests per Minute')
    ax.set_title('Throughput Over Time', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    avg_throughput = throughput['requests_per_min'].mean()
    ax.axhline(y=avg_throughput, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_throughput:.1f} req/min')
    ax.legend()


def plot_percentile_chart(df, ax):
    percentiles = [50, 75, 90, 95, 99, 99.9]
    values = [df['response_time'].quantile(p/100) for p in percentiles]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(percentiles)))
    bars = ax.bar([f'{p}th' for p in percentiles], values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title('Response Time Percentiles', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold')


def plot_active_threads_over_time(df, ax):
    ax.plot(df['start_time'], df['active_threads'], linewidth=2, color='purple', alpha=0.7)
    ax.fill_between(df['start_time'], 0, df['active_threads'], alpha=0.3, color='purple')
    ax.set_xlabel('Time')
    ax.set_ylabel('Active Threads')
    ax.set_title('Active Threads Over Time', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def plot_status_code_distribution(df, ax):
    status_counts = df['response_status_code'].value_counts().sort_index()
    colors = ['green' if code == 200 else 'orange' if code < 400 else 'red' for code in status_counts.index]
    bars = ax.bar(status_counts.index.astype(str), status_counts.values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('HTTP Status Code')
    ax.set_ylabel('Count')
    ax.set_title('HTTP Status Code Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, status_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val}', ha='center', va='bottom', fontweight='bold')


def plot_error_vs_success_pie(df, ax):
    error_counts = df['is_error'].value_counts()
    labels = ['Success', 'Error']
    values = [error_counts.get(0, 0), error_counts.get(1, 0)]
    colors = ['#90EE90', '#FF6B6B']
    explode = (0, 0.1)
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    ax.set_title('Success vs Error Rate', fontsize=14, fontweight='bold')


def create_cover_page(ax, project_name="ChatMFC", done_by="QA Team", test_date=None):
    """Create a professional cover page with project details"""
    ax.axis('off')

    if test_date is None:
        test_date = datetime.now().strftime('%B %d, %Y')

    # Main title
    ax.text(0.5, 0.75, 'BENCH TEST SUMMARY',
            ha='center', va='center', fontsize=32, fontweight='bold',
            transform=ax.transAxes, color='#2C3E50')

    # Subtitle line
    ax.plot([0.2, 0.8], [0.68, 0.68], 'k-', linewidth=2, transform=ax.transAxes)

    ax.text(0.5, 0.63, 'Performance Test Report',
            ha='center', va='center', fontsize=18, style='italic',
            transform=ax.transAxes, color='#34495E')

    # Project details box
    box_props = dict(boxstyle='round,pad=0.8', facecolor='#E8F4F8', edgecolor='#3498DB', linewidth=2)

    details_text = f"""Project Name: {project_name}

Done By: {done_by}

Test Date: {test_date}"""

    ax.text(0.5, 0.40, details_text,
            ha='center', va='center', fontsize=16,
            transform=ax.transAxes, bbox=box_props,
            family='monospace', linespacing=2.0)

    # Footer
    ax.text(0.5, 0.15, 'Automated Performance Testing Framework',
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes, color='#7F8C8D', style='italic')

    ax.text(0.5, 0.10, '© 2026 Performance Test Suite',
            ha='center', va='center', fontsize=9,
            transform=ax.transAxes, color='#95A5A6')


def create_summary_table(stats, ax):
    """Create performance metrics summary table"""
    ax.axis('tight')
    ax.axis('off')

    # Performance metrics section
    table_data = []
    for key, value in stats.items():
        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
        table_data.append([key, formatted_value])

    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # Style column headers
    for i in range(2):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

    # Style performance metrics with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
            # Make metric names bold
            if j == 0:
                table[(i, j)].set_text_props(weight='bold')

    ax.set_title('Performance Metrics Summary', fontsize=18, fontweight='bold', pad=20, color='#2C3E50')



def generate_pdf_report(db_path=None, output_path=None, project_name="ChatMFC", done_by="QA Team"):
    """Generate comprehensive PDF report with all graphs"""
    try:
        if db_path is None:
            db_path = get_latest_db_file()

        df = load_data_from_db(db_path)

        if len(df) == 0:
            print("No data found in database. Cannot generate report.")
            return

        stats = create_summary_stats(df)

        if output_path is None:
            db_name = os.path.basename(db_path).replace('.db', '')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'./reports/performance_report_{db_name}_{timestamp}.pdf'

        os.makedirs('./reports', exist_ok=True)

        test_date = datetime.now().strftime('%B %d, %Y')

        print(f"\nGenerating PDF report: {output_path}")
        print("=" * 80)

        with PdfPages(output_path) as pdf:
            # Page 1: Cover Page
            print("Creating page 1: Cover Page...")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            create_cover_page(ax, project_name, done_by, test_date)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            # Page 2: Performance Metrics Summary
            print("Creating page 2: Performance Metrics Summary...")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            create_summary_table(stats, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 3: Response Time Over Time...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_response_time_over_time(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 4: Response Time Distribution...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_response_time_distribution(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 5: Response Time vs Concurrent Users...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_response_time_by_users(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 6: Response Time Percentiles...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_percentile_chart(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 7: Error Rate Over Time...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_error_rate_over_time(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 8: Throughput Over Time...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_throughput_over_time(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 9: Active Threads Over Time...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_active_threads_over_time(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 10: Status Code Distribution...")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_status_code_distribution(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 11: Success vs Error Rate...")
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_error_vs_success_pie(df, ax)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            d = pdf.infodict()
            d['Title'] = 'Performance Test Report'
            d['Author'] = 'Performance Test Framework'
            d['Subject'] = 'Benchmark Test Results'
            d['Keywords'] = 'Performance Testing, Load Testing, Benchmarking'
            d['CreationDate'] = datetime.now()

        print("=" * 80)
        print(f"\n✓ PDF report generated successfully!")
        print(f"  Location: {output_path}")
        print(f"  Total pages: 11")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
        print("\nSummary Statistics:")
        print("-" * 80)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 80 + "\n")

        return output_path

    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        generate_pdf_report(db_path, output_path)
    else:
        generate_pdf_report()
