project_name="MyProject"
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

suite_id=""
test_name=""
test_start_time=None
error_percent=0
running_users=0
current_errors=0

standard_sla = {
  "avg": 0.05,
  "percentile_10": 0.1,
  "percentile_20": 0.2,
  "percentile_30": 0.3,
  "percentile_40": 0.4,
  "percentile_50": 0.5,
  "percentile_60": 0.6,
  "percentile_70": 0.7,
  "percentile_80": 0.8,
  "percentile_90": .9,
  "percentile_95": .95,
  "percentile_99": .99
}

project_sla={
  "avg": 1.56,
  "percentile_10": 0.5,
  "percentile_20": 0.6,
  "percentile_30": 0.7,
  "percentile_40": 0.8,
  "percentile_50": 0.9,
  "percentile_60": 1.0,
  "percentile_70": 1.2,
  "percentile_80": 1.5,
  "percentile_90": 2.0,
  "percentile_95": 3.0,
  "percentile_99": 5.0
}
