import time

print(time.time())
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(time.localtime())

start = time.perf_counter()
time.sleep(1)
end = time.perf_counter()
print(end - start)
print(int((end - start)*1000))