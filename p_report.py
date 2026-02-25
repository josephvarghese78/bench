"""
Performance Test Report Generator
Generates a comprehensive PDF report with graphs from benchmark test results
pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pyodbc
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
#from pyodbc import connect
import config as cfg
from reportlab.lib import pagesizes, colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
conn=None

def get_latest_db_file(db_folder="./db"):
    """Get the most recent database file"""
    db_files = glob.glob(f"{db_folder}/*.db")
    print(db_files)
    if not db_files:
        raise FileNotFoundError(f"No database files found in {db_folder}")
    latest_db = max(db_files, key=os.path.getctime)
    print("Latest database file found:", latest_db)
    return latest_db



def connect_to_db(db_path):
    global conn
    conn = sqlite3.connect(db_path)

def close_db():
    global conn
    if conn:
        conn.close()
        conn = None

def get_id_data():
    query="""
    select distinct suite_name, max_threads  from performance
    """
    #suite_names = [row[0] for row in conn.execute(query)]
    rows = conn.execute(query).fetchall()
    return rows

def load_test_data(suite_name=None):
    """Load performance data from SQLite database"""
    global conn


    query = f"""
    SELECT suite_name, test_name, max_threads, thread_name, iteration,
           start_time, end_time, samples_started, samples_completed,
           active_threads, think_time, response_time, error_status,
           error_percent, response_status_code, response
    FROM performance where suite_name = '{suite_name}'
    ORDER BY start_time
    """

    df = pd.read_sql_query(query, conn)

    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    df['is_error'] = df['error_status'].apply(lambda x: 1 if x == 'F' else 0)

    print(f"Loaded {len(df)} records")
    return df

def latency_data():
    """Fetch performance test data from SQLite into a pandas DataFrame"""
    #conn = sqlite3.connect(db_path)
    global conn

    query="""
    WITH ordered AS (
    SELECT suite_name, test_name, response_time/1000 as response_time, max_threads,
           ROW_NUMBER() OVER (PARTITION BY suite_name, test_name ORDER BY response_time/1000) AS rn,
           COUNT(*) OVER (PARTITION BY suite_name, test_name) AS cnt
        FROM performance
    )
    SELECT suite_name, max_threads, test_name,
           MIN(response_time) AS min,
           MAX(response_time) AS max,
           AVG(response_time) AS avg,
           max_threads/AVG(response_time) as throughput,
           (max_threads/AVG(response_time))/max_threads as efficiency,
           (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = (cnt + 1)/2) AS median_response_time,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.1 * cnt AS INT)) AS percentile_10,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.2 * cnt AS INT)) AS percentile_20,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.3 * cnt AS INT)) AS percentile_30,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.4 * cnt AS INT)) AS percentile_40,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.5 * cnt AS INT)) AS percentile_50,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.6 * cnt AS INT)) AS percentile_60,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.7 * cnt AS INT)) AS percentile_70,
            (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.8 * cnt AS INT)) AS percentile_80,
           (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.9 * cnt AS INT)) AS percentile_90,
           (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.95 * cnt AS INT)) AS percentile_95,
           (SELECT response_time FROM ordered o2 
            WHERE o2.suite_name = o1.suite_name AND o2.test_name = o1.test_name AND rn = CAST(0.99 * cnt AS INT)) AS percentile_99
    FROM ordered o1
    GROUP BY suite_name, max_threads, test_name
    order by max_threads 
    """
    df = pd.read_sql_query(query, conn)
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


def plot_response_time_over_time(users, df, ax):
    ax.plot(df['start_time'], df['response_time'], alpha=0.6, linewidth=1, color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title(f'Response Time Over Time ({users} user test)', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    mean_rt = df['response_time'].mean()
    ax.axhline(y=mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}ms')
    ax.legend()


def plot_response_time_distribution(users, df, ax):
    ax.hist(df['response_time'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Response Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Response Time Distribution ({users} user test)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    mean_rt = df['response_time'].mean()
    median_rt = df['response_time'].median()
    p95 = df['response_time'].quantile(0.95)
    ax.axvline(x=mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}ms')
    ax.axvline(x=median_rt, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rt:.2f}ms')
    ax.axvline(x=p95, color='orange', linestyle='--', linewidth=2, label=f'95th %ile: {p95:.2f}ms')
    ax.legend()


def plot_response_time_by_users(users, df, ax):
    grouped = df.groupby('max_threads').agg({'response_time': ['mean', 'median', 'min', 'max']}).reset_index()
    grouped.columns = ['max_threads', 'mean_rt', 'median_rt', 'min_rt', 'max_rt']
    ax.plot(grouped['max_threads'], grouped['mean_rt'], marker='o', linewidth=2, markersize=8, label='Mean', color='blue')
    ax.plot(grouped['max_threads'], grouped['median_rt'], marker='s', linewidth=2, markersize=8, label='Median', color='green')
    ax.fill_between(grouped['max_threads'], grouped['min_rt'], grouped['max_rt'], alpha=0.2, color='gray', label='Min-Max Range')
    ax.set_xlabel('Number of Concurrent Users')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title(f'Response Time vs Concurrent Users ({users} user test)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_error_rate_over_time(users, df, ax):
    df_sorted = df.sort_values('start_time')
    window_size = max(1, len(df) // 50)
    df_sorted['window'] = df_sorted.index // window_size
    error_rate = df_sorted.groupby('window').agg({'is_error': 'mean', 'start_time': 'first'}).reset_index()
    error_rate['error_rate_pct'] = error_rate['is_error'] * 100
    ax.plot(error_rate['start_time'], error_rate['error_rate_pct'], linewidth=2, color='red', marker='o', markersize=4)
    ax.fill_between(error_rate['start_time'], 0, error_rate['error_rate_pct'], alpha=0.3, color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title(f'Error Rate Over Time ({users} user test)', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def plot_throughput_over_time(users, df, ax):
    df_sorted = df.sort_values('start_time')
    df_sorted['time_minute'] = df_sorted['start_time'].dt.floor('1min')
    throughput = df_sorted.groupby('time_minute').size().reset_index(name='requests_per_min')
    ax.bar(throughput['time_minute'], throughput['requests_per_min'], width=0.0005, color='green', alpha=0.7, edgecolor='darkgreen')
    ax.set_xlabel('Time')
    ax.set_ylabel('Requests per Minute')
    ax.set_title(f'Throughput Over Time ({users} user test)', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    avg_throughput = throughput['requests_per_min'].mean()
    ax.axhline(y=avg_throughput, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_throughput:.1f} req/min')
    ax.legend()


def plot_percentile_chart(users, df, ax):
    percentiles = [10,20,30,40, 50, 60, 70, 80, 90, 95, 99, 99.9]
    values = [df['response_time'].quantile(p/100) for p in percentiles]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(percentiles)))
    bars = ax.bar([f'{p}th' for p in percentiles], values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title(f'Response Time Percentiles ({users} user test)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold')


def plot_active_threads_over_time(users, df, ax):
    ax.plot(df['start_time'], df['active_threads'], linewidth=2, color='purple', alpha=0.7)
    ax.fill_between(df['start_time'], 0, df['active_threads'], alpha=0.3, color='purple')
    ax.set_xlabel('Time')
    ax.set_ylabel('Active Threads')
    ax.set_title(f'Active Threads Over Time ({users} user test)', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def plot_status_code_distribution(users, df, ax):
    status_counts = df['response_status_code'].value_counts().sort_index()
    colors = ['green' if code == 200 else 'orange' if code < 400 else 'red' for code in status_counts.index]
    bars = ax.bar(status_counts.index.astype(str), status_counts.values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('HTTP Status Code')
    ax.set_ylabel('Count')
    ax.set_title(f'HTTP Status Code Distribution ({users} user test)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, status_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{val}', ha='center', va='bottom', fontweight='bold')


def plot_error_vs_success_pie(users, df, ax):
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
    ax.set_title(f'Success vs Error Rate ({users} user test)', fontsize=14, fontweight='bold')


def create_summary_page(ax, result):

    ax.axis('off')
    ax.set_title("Conclusion", fontsize=14, fontweight='bold')
    ax.text(0, 0.75, f'System State: {result["state"]}', fontsize=10)
    ax.text(0, 0.70, f'Safe Concurrency: {result["safe_users"]}', fontsize=10)
    ax.text(0, 0.65, f'Safe Throughput: {result["safe_throughput"]}', fontsize=10)

    if result["saturation_users"] is None:
        ax.text(0, .60, f'Saturation found: NA', fontsize=10)
    else:
        ax.text(0, .55, f'Saturation Starts At: {result["saturation_users"]}', fontsize=10)

    ax.text(0, .50, f'**Notes:', fontsize=10)
    v=.45
    if len(result["notes"])>0:
        for note in result["notes"]:
            ax.text(0, v, f'*******>: {note}', fontsize=8)
            v-=.05
    else:
        ax.text(0, v, f'*******>: NA', fontsize=8)





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



def create_user_latency_report(ax, df, sla, type):
    percentile_cols = list(sla.keys())[1:]

    ax.axis('off')  # no axes for table
    ax.set_title(f'Heat MAP - {type}', fontsize=18, fontweight='bold', pad=20,
                 color='#2C3E50')

    # Build table data for matplotlib
    header = ["Users", "Avg"] + [f'{p.split("_")[1]}%' for p in percentile_cols]
    table_data = [header]
    for _, row in df.iterrows():
        row_data = [row["max_threads"]] + [f'{row["avg"]:.3f}']+ [f"{row[p]:.3f}" for p in percentile_cols]
        table_data.append(row_data)

    # Create matplotlib table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')

    # Style table (optional)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)  # adjust row height

    # Optional: color cells based on SLA
    for row_idx, row in enumerate(df.itertuples(), start=1):
        for col_idx, col in enumerate(['avg'] + percentile_cols, start=1):
            value = getattr(row, col)
            color = "green" if value <= sla[col] else "red"
            table[(row_idx, col_idx)].set_text_props(color=color)



def create_latency_trend_report(ax, df, col_name, title):

    ax.plot(df['max_threads'], df[col_name], alpha=0.6, linewidth=1, color='blue')
    ax.set_xlabel('Users')
    ax.set_ylabel("Response Time (ms)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    mean_rt = df[col_name].mean()

    #ticks=np.linspace(df[col_name].min(), df[col_name].max(), 5)
    #ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter('{x:.3f}')
    ax.axhline(y=mean_rt, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.2f}ms')
    ax.legend()

def create_userscore_table(ax, items):
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for item in items:
        table_row=[]
        for key,value in item.items():
            formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            table_row.append(formatted_value)

        table_data.append(table_row)

    table = ax.table(cellText=table_data, colLabels= ["Users","Latency-SLA","Throughput-Stablity","Latency-Stability","Efficiency-Drop",
        "Weighted-Score","Grade"], cellLoc='left', loc='center', colWidths=[0.1, 0.2, 0.3, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)


    # Style column headers
    for i in range(7):
        table[(0, i)].set_facecolor('#3498DB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)


    # Style performance metrics with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(7):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
            # Make metric names bold
            #if j == 0:
            #    table[(i, j)].set_text_props(weight='bold')

    ax.set_title(f'Test Score & Grade', fontsize=18, fontweight='bold', pad=20,
                 color='#2C3E50')

def create_summary_table(users, stats, ax):
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

    ax.set_title(f'Performance Metrics Summary ({users} user test.)', fontsize=18, fontweight='bold', pad=20, color='#2C3E50')


def detect_saturation(
        threads,
        avg_latency,
        latency_threshold=0.10,  # 10% latency growth allowed between steps
        efficiency_drop_threshold=0.10,  # 10% efficiency drop allowed
        throughput_flat_threshold=0.05,  # 5% throughput gain considered flat
        max_latency_sla=1.0  # absolute SLA threshold in seconds
):

    sum_of_threads=np.sum(threads)
    min_thread=np.min(threads)
    max_thread=np.max(threads)

    threads = np.array(threads, dtype=float)
    avg_latency = np.array(avg_latency, dtype=float)

    # Metrics
    throughput = threads / avg_latency
    efficiency = throughput / threads  # per-user throughput

    # Derivatives
    throughput_gain_pct = np.diff(throughput) / throughput[:-1]
    latency_growth = np.diff(avg_latency) / avg_latency[:-1]
    efficiency_drop = -np.diff(efficiency) / efficiency[:-1]

    safe_index = 0
    saturation_index = None
    notes = []
    test_score=[]
    test_score_data={}

    for i in range(0, len(threads)):
        test_score_data={}
        test_score_data["users"] = str(int(threads[i]))
        test_score_data["latency_sla"]=0
        test_score_data["throughput_stablity"]=0
        test_score_data["latency_stability"]=0
        test_score_data["efficiency_drop"]=0

        load_factor=(threads[i]-min_thread)/(max_thread-min_thread)
        inverse_load = 1 - load_factor
        risk_factor = 1 + (1.3 * inverse_load) + (0.7 * load_factor)

        # Check absolute SLA
        sla_violation = avg_latency[i] > max_latency_sla
        if sla_violation:
            saturation_index = i
            test_score_data["latency_sla"] = 40 * risk_factor
            notes.append(f"Latency {avg_latency[i]:.3f}s exceeds SLA {max_latency_sla}s")
            #break

        # Healthy scaling check
        tp_health = throughput_gain_pct[i - 1] > throughput_flat_threshold
        lt_health= latency_growth[i - 1] < latency_threshold
        ef_health= efficiency_drop[i - 1] < efficiency_drop_threshold
        healthy=(tp_health and lt_health and ef_health)

        if healthy:
            safe_index = i
        else:
            saturation_index = i
            if not tp_health:
                test_score_data["throughput_stablity"] = 25 * (1 + 0.7 * (risk_factor - 1))
                notes.append(f"Throughput flattening reducing overtime, this indicates system is failing to scale or not getting response in time, found in {i} user test.")

            if not lt_health:
                test_score_data["latency_stability"] = 25 * (1 + 0.5 * (risk_factor - 1))
                notes.append(f"Lattency is increasing, this indicates when load increases response time delays, found in {i} user test.")

            if not ef_health:
                test_score_data["efficiency_drop"] = 10 * 1
                notes.append(f"Efficiency drops >10% when user load increases, found in {i} user test.")
            break
        test_score_data["weighted_score"]=100-(test_score_data["latency_sla"]+
                                                            test_score_data["throughput_stablity"]+
                                                            test_score_data["latency_stability"]+
                                                            test_score_data["efficiency_drop"])

        if test_score_data["weighted_score"]>=90 and test_score_data["weighted_score"]<=100:
            test_score_data["grade"]="Excellent"
        elif test_score_data["weighted_score"]>=80 and test_score_data["weighted_score"]<=89:
            test_score_data["grade"]="Very Good"
        elif test_score_data["weighted_score"]>=70 and test_score_data["weighted_score"]<=79:
            test_score_data["grade"]="Good"
        elif test_score_data["weighted_score"]>=60 and test_score_data["weighted_score"]<=69:
            test_score_data["grade"]="Fair"
        elif test_score_data["weighted_score"]>=50 and test_score_data["weighted_score"]<=59:
            test_score_data["grade"]="Poor"
        else:
            test_score_data["grade"]="Fail"

        test_score.append(test_score_data)

    state = "SCALING (No saturation detected)" if saturation_index is None else "SATURATION DETECTED"

    return {
        "safe_users": int(threads[safe_index]),
        "safe_throughput": float(throughput[safe_index]),
        "saturation_users": int(threads[saturation_index]) if saturation_index is not None else None,
        "state": state,
        "throughput": throughput,
        "efficiency": efficiency,
        "avg_latency":avg_latency,
        "notes": notes,
        "threads": threads,
        "test_score": test_score
    }


def generate_pdf_report(db_path=None, output_path=None, project_name="ChatMFC", done_by="QA Team"):
    """Generate comprehensive PDF report with all graphs"""
    try:
        db_path = get_latest_db_file(db_path)
        connect_to_db(db_path)
        user_threads=[]



        #if output_path is None:
        db_name = os.path.basename(db_path).replace('.db', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'./{output_path}/performance_report_{db_name}_{timestamp}.pdf'

        os.makedirs('./reports', exist_ok=True)

        test_date = datetime.now().strftime('%B %d, %Y')

        print(f"\nGenerating PDF report: {output_path}")
        print("=" * 80)

        with PdfPages(output_path) as pdf:
            rows=get_id_data()



            # Page 1: Cover Page
            print("Creating page 1: Cover Page...")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            create_cover_page(ax, project_name, done_by, test_date)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()


            for suite, users in rows:
                user_threads.append(users)
                df = load_test_data(suite)
                stats = create_summary_stats(df)

                # Page 2: Performance Metrics Summary
                print("Creating page 2: Performance Metrics Summary...")
                fig, ax = plt.subplots(figsize=(11, 8.5))
                create_summary_table(users, stats, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()


                print("Creating page 3: Response Time Over Time...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_response_time_over_time(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 4: Response Time Distribution...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_response_time_distribution(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 5: Response Time vs Concurrent Users...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_response_time_by_users(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 6: Response Time Percentiles...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_percentile_chart(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 7: Error Rate Over Time...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_error_rate_over_time(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 8: Throughput Over Time...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_throughput_over_time(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 9: Active Threads Over Time...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_active_threads_over_time(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 10: Status Code Distribution...")
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_status_code_distribution(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                print("Creating page 11: Success vs Error Rate...")
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_error_vs_success_pie(users, df, ax)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

                d = pdf.infodict()
                d['Title'] = 'Performance Test Report'
                d['Author'] = 'Performance Test Framework'
                d['Subject'] = 'Benchmark Test Results'
                d['Keywords'] = 'Performance Testing, Load Testing, Benchmarking'
                d['CreationDate'] = datetime.now()

            df_latency = latency_data()
            print("Creating page 12: Heat MAP... standard")
            fig, ax = plt.subplots(figsize=(12, 6))
            create_user_latency_report(ax, df_latency, cfg.standard_sla, "Standard")
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 13: Heat MAP.. project")
            fig, ax = plt.subplots(figsize=(12, 6))
            create_user_latency_report(ax, df_latency, cfg.project_sla, cfg.project_name)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()




            print("Creating page 14: Latency/Pecentile trend")
            cols=['Avg', '90','95','99']
            for col in cols:
                if col=="Avg":
                    col_name="avg"
                    title="Avg Latency Trend"
                else:
                    col_name=f"percentile_{col}"
                    title= f"{col}% Latency Trend "
                fig, ax = plt.subplots(figsize=(12, 6))
                create_latency_trend_report(ax, df_latency, col_name, title)
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()

            print("Creating page 15: Throughput trend")
            fig, ax = plt.subplots(figsize=(12, 6))
            create_latency_trend_report(ax, df_latency, "throughput", "Throughput Trend")
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 15: Efficiency trend")
            fig, ax = plt.subplots(figsize=(12, 6))
            create_latency_trend_report(ax, df_latency, "efficiency", "Efficiency Trend")
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            threads = []
            avg_lat = []
            for _, row in df_latency.iterrows():
                threads.append(row["max_threads"])
                avg_lat.append(row["avg"])

            result = detect_saturation(threads, avg_lat)

            print("Creating page 16: User Score & Grade")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            create_userscore_table(ax, result["test_score"])
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()

            print("Creating page 17: Summary")
            fig, ax = plt.subplots(figsize=(11, 8.5))
            #create_latency_trend_report(ax, df_latency, "efficiency", "Efficiency Trend")
            create_summary_page(ax, result)
            plt.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()




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


generate_pdf_report("./db", "./reports", "test123", "QA Team")


#create_userscore_table(None, None)