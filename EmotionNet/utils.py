import time

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        elap = time.time() - ts
 
        hr = elap // 3600
        minsec = elap % 3600
        min = minsec // 60
        sec = minsec % 60 

        print('func:%r args:[%r, %r] took: %2d:%02d:%06.3f' % (f.__name__, args, kw, hr, min, sec))
        return result

    return timed

