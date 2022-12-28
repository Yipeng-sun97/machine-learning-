from threading import Thread

def f():
    while True:
        b = 53*53

if __name__ == '__main__':
    plist = []
    # 启动10个线程
    for i in range(10):
        p = Thread(target=f)
        p.start()
        plist.append(p)

    for p in plist:
        p.join()