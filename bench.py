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


try:
    with open(f"./data.json", 'r') as f:
        cfg.requests_data = json.load(f)
except:
    pass
cfg.project_name=cfg.requests_data.get("project","Project")
count = sum(1 for f in os.listdir("./db") if f.startswith(cfg.project_name) and f.endswith("db"))
cfg.db_filename = f"{cfg.project_name}.db" if count==0 else f"{cfg.project_name}_{count}.db"
shutil.copy(f"./templates/template.db", f"./db/{cfg.db_filename}")
#db_lock = threading.Lock()


def log_performance(test_name,
    thread_name, iteration, start_time, end_time, think_time,
    response_time, error_status, error_percent, response_status_code, response):

    db_conn = sqlite3.connect(f"./db/{cfg.db_filename}")
    db_cursor = db_conn.cursor()

    #with db_lock:
    db_cursor.execute("""
            INSERT INTO performance (suite_name,
            test_name, max_threads, thread_name, iteration, start_time,
            end_time, think_time, response_time, error_status, error_percent, response_status_code,
            response ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?,?)
            """,
        (cfg.suite_id, test_name, cfg.users, thread_name, iteration, start_time, end_time,
         think_time, response_time, error_status, error_percent, response_status_code,response ))

    db_conn.commit()
    db_conn.close()




def apirequest(benchtest):
    """
    Perform API requests defined in cfg.requests_data['requests'].
    Returns:
        resp_content: response content (text or JSON)
        status_code: HTTP status code
        error_flag: 0 if success, 1 if error
        think_time: random sleep before request
    """
    resp_content = None
    status_code = 0
    error_flag = 0

    # Choose random think time from cfg
    tt = 0
    test_name=""

    try:

        test_name = benchtest.get("name","")
        method = benchtest.get("method", "get").lower()
        url = benchtest.get("url")
        headers = benchtest.get("headers", {})
        verify = benchtest.get("verify", True)
        timeout = benchtest.get("timeout", 30)
        data = benchtest.get("data", None)  # for POST/PUT payload
        files=benchtest.get("files", None)

        tt = random.choice(cfg.think_time)

        # Simulate think time
        time.sleep(tt)

        # Perform request
        if method == "get":
            resp = requests.get(url, headers=headers, timeout=timeout, verify=verify)
        elif method == "post":
            resp = requests.post(url, headers=headers, json=data, timeout=timeout, verify=verify)
        elif method == "put":
            resp = requests.put(url, headers=headers, json=data, timeout=timeout, verify=verify)
        elif method == "delete":
            resp = requests.delete(url, headers=headers, timeout=timeout, verify=verify)
        else:
            # Unsupported method
            return None, 0, 1, tt, test_name

        # Check if response is JSON
        try:
            resp_content = resp.json()
        except ValueError:
            resp_content = resp.text

        status_code = resp.status_code
        error_flag = 0 if resp.ok else 1

        return resp_content, status_code, error_flag, tt, test_name

    except requests.exceptions.RequestException as e:
        # Network or timeout error
        resp_content = str(e)
        status_code = 0
        error_flag = 1
        return resp_content, status_code, error_flag, tt, test_name

    except Exception as e:
        return str(e), status_code, 1, tt, test_name



def sample(threadname, benchtest ):
    #print(time.time(), cfg.test_start_time, time.time() - cfg.test_start_time, cfg.runfor*60)
    i=0
    while (time.time() - cfg.test_start_time) < (cfg.runfor*60):
        if cfg.error_percent >=cfg.error_treshold:
            break
        i+=1
        cfg.samples+=1
        start_time = datetime.now()
        start_time_pc=time.perf_counter()
        resp, resp_code, status, think_time, test_name = apirequest(benchtest)
        end_time = datetime.now()
        end_time_pc=time.perf_counter()
        response_time = (end_time_pc - start_time_pc)*1000


        if status==1:
            cfg.current_errors+=1

        cfg.error_percent=round((cfg.current_errors/cfg.samples)*100,0)
        print(threadname, resp_code)

        log_performance(test_name, threadname, i,
                        start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        think_time*1000,
                        response_time,
                        status,
                        cfg.error_percent,
                        resp_code,
                        str(resp))



def runtest():
    benchtests = cfg.requests_data.get("benchtests", [])
    for benchtest in benchtests:
        cfg.users=benchtest['testparams'].get("users", 0)
        cfg.user_step=benchtest['testparams'].get("user_step", -2)
        cfg.runfor=benchtest['testparams'].get("runfor", 5)
        cfg.valid_status_codes=benchtest['testparams'].get("valid_status_codes", [200])
        cfg.error_treshold=benchtest['testparams'].get("error_treshold", 5)
        cfg.think_time=benchtest['testparams'].get("think_time", [2,4,6])
        cfg.rampup=benchtest['testparams'].get("rampup", 30)
        delay_between_thread = cfg.rampup/(cfg.users-1)

        while cfg.users>0:
            cfg.suite_id=str(uuid.uuid4())
            cfg.test_start_time = time.time()
            cfg.current_errors=0
            cfg.samples=0
            cfg.running_users=0
            cfg.error_percent=0

            threads=[]
            thread_name=""
            i=0
            print("suite_id", cfg.suite_id)
            print("users", cfg.users)
            for _ in range(cfg.users):
                i += 1
                cfg.running_users=i
                thread_name=f"User-{i}"
                t = threading.Thread(target=sample, args=(thread_name, benchtest,))
                threads.append(t)
                t.start()
                if i< cfg.users:
                    time.sleep(delay_between_thread)

            for t in threads:
                t.join()
            print("error%", cfg.error_percent)

            if cfg.error_percent >= cfg.error_treshold:
                cfg.users+=cfg.user_step
            else:
                cfg.users+=cfg.user_step
                #break



runtest()
print("Test completed!, logs in: ", cfg.db_filename)

