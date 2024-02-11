import requests
import json
import datetime
import time
import yaml

t_now = datetime.datetime.now()
t_9 = t_now.replace(hour=9, minute=0, second=0, microsecond=0)
t_start = t_now.replace(hour=9, minute=5, second=0, microsecond=0)
t_sell = t_now.replace(hour=15, minute=15, second=0, microsecond=0)
t_exit = t_now.replace(hour=15, minute=20, second=0,microsecond=0)
today = datetime.datetime.today().weekday()


print("t_now",t_now.strftime("%Y%m%d"));