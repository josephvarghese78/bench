valid_status_codes=[200,201,204]
ignore_status_codes=[429,499]
error_threshold=5
users=10
runfor=5
user_step=5
think_time=[4, 6, 8]
rampup_per_user=[2,4,6,8,10]
delay_between_thread=1
stop_at_user=50

error_flags=['F']  # P=Pass, F=Fail, W=

#active_threads=0
samples_started=0
samples_completed=0

requests_data=None
db_conn=None
db_filename=""
db_cursor=None

project_name=""
suite_id=""
test_name=""
test_start_time=None
error_percent=0
running_users=0
current_errors=0
