import random
import threading
import os, json
import requests
import config as cfg
import shutil
import datetime as dt
from datetime import datetime
import sqlite3
import uuid
import time
import random

import os
import re

import os
import re
import custom as custcode

try:
    with open(f"./data.json", 'r') as f:
        cfg.requests_data = json.load(f)
except:
    pass
cfg.project_name = cfg.requests_data.get("project", "Project")
count = sum(1 for f in os.listdir("./db") if f.startswith(cfg.project_name) and f.endswith("db"))
cfg.db_filename = f"{cfg.project_name}.db" if count == 0 else f"{cfg.project_name}_{count}.db"
shutil.copy(f"./templates/template.db", f"./db/{cfg.db_filename}")


def log_performance(test_name,
                    thread_name, iteration, start_time, end_time,
                    samples_started, samples_completed, active_threads,think_time,
                    response_time, error_status, error_percent, response_status_code, response):
    db_conn = sqlite3.connect(f"./db/{cfg.db_filename}")
    db_cursor = db_conn.cursor()

    db_cursor.execute("""
            INSERT INTO performance (suite_name,
            test_name, max_threads, thread_name, iteration, start_time,
            end_time, samples_started, samples_completed, active_threads,
            think_time, response_time, error_status, error_percent, response_status_code,
            response ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?,?,?,?,?)
            """,
                      (cfg.suite_id, test_name, cfg.users, thread_name, iteration, start_time, end_time,
                       samples_started, samples_completed, active_threads,
                       think_time, response_time, error_status, error_percent, response_status_code, response))

    db_conn.commit()
    db_conn.close()


class user_thread():

    thread_name = ""

    def start_user(self):
        i=0
        user_session = requests.session()
        while (time.time() - cfg.test_start_time) < ((cfg.runfor+0) * 60):
            if cfg.error_percent >= cfg.error_threshold:
                print(f"[{self.thread_name}] Error threshold reached, stopping...")
                break
            i += 1
            cfg.samples_started += 1

            resp, status_code, error_flag, think_time, test_name, start_time, start_time_pc, end_time, end_time_pc, response_time = custcode.api_request_main(user_session)
            cfg.samples_completed+=1
            if error_flag in cfg.error_flags:
                cfg.current_errors += 1
                print(f"\n[{self.thread_name}] ❌ Request failed")
                print(f"  Status Code: {status_code}")
                print(f"  Response Time: {response_time:.2f}ms")
            else:
                print(f"\n[{self.thread_name}] ✓ Request successful")
                print(f"  Status Code: {status_code}")
                print(f"  Response Time: {response_time:.2f}ms")

            cfg.error_percent = (cfg.current_errors / cfg.samples_started) * 100
            print(f"\n[Summary]")
            print(f"  Users: {cfg.users}")
            print(f"  Thread: {self.thread_name}")
            print(f"  Start Time: {cfg.test_start_time}")
            print(f"  Iteration: {i}")
            print(f"  Status_Code: {status_code}")
            print(f"  Error Rate: {cfg.error_percent}%")
            print(f"  Total Samples: {cfg.samples_started}")
            print("-" * 80)

            log_performance(test_name, self.thread_name, i,
                            start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            cfg.samples_started,
                            cfg.samples_completed,
                            cfg.running_users,
                            think_time * 1000,
                            response_time,
                            error_flag,
                            cfg.error_percent,
                            status_code,
                            str(resp))
        if (time.time() - cfg.test_start_time) >= ((cfg.runfor+0) * 60):
            cfg.running_users-=1


def perftest(thread_name):
    user = user_thread()
    user.thread_name = thread_name
    user.start_user()




def runtest():
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST STARTED")
    print(f"Project: {cfg.project_name} | Database: {cfg.db_filename}")
    print("=" * 80 + "\n")

    while cfg.users > 0 and cfg.users < cfg.stop_at_user:
        cfg.suite_id = str(uuid.uuid4())
        cfg.test_start_time = time.time()
        cfg.current_errors = 0
        cfg.samples_started = 0
        cfg.samples_completed = 0
        cfg.running_users = 0
        cfg.error_percent = 0
        #cfg.delay_between_thread = cfg.rampup / (cfg.users - 1)



        threads = []
        thread_name = ""
        i = 0

        print("=" * 80)
        print(f"Test Suite ID: {cfg.suite_id}")
        print(f"Configuration: Users={cfg.users}, Run Duration={cfg.runfor}min, Error Threshold={cfg.error_threshold}%")
        #print(f"Ramp-up Time: {ramp_up:.2f}s")
        print("=" * 80 + "\n")

        for _ in range(cfg.users):
            i += 1
            cfg.running_users = i
            thread_name = f"User-{i}"
            #user_session = requests.session()
            t = threading.Thread(target=perftest, args=(thread_name,))
            threads.append(t)
            t.start()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Started {thread_name}")
            if i < cfg.users:
                if isinstance(cfg.rampup_per_user, list):
                    time.sleep(random.choice(cfg.rampup_per_user))
                elif isinstance(cfg.rampup_per_user, int):
                    time.sleep(cfg.rampup_per_user)
                else:
                    time.sleep(2)


        print(f"\n[INFO] All {cfg.users} threads started successfully, waiting for completion...\n")

        for t in threads:
            t.join()

        print("\n" + "=" * 80)
        print(f"Test Suite Completed | Final Error Rate: {cfg.error_percent}%")
        print(f"Total Samples: {cfg.samples_started} | Total Errors: {cfg.current_errors}")
        print("=" * 80 + "\n")

        if cfg.error_percent >= cfg.error_threshold:
            print("⚠️  ERROR THRESHOLD REACHED - Test stopped!")
            print(f"Maximum acceptable error rate ({cfg.error_threshold}%) exceeded.")
            break
        else:
            cfg.users += cfg.user_step
            if cfg.users > 0:
                print(f"➡️  Scaling up to {cfg.users} users for next iteration...\n")


runtest()
print("\n" + "=" * 80)
print("✓ TEST COMPLETED SUCCESSFULLY!")
print(f"Results saved to: ./db/{cfg.db_filename}")
print("=" * 80 + "\n")