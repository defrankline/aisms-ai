from datetime import date, timedelta


def month_start(d: date) -> date:
    return d.replace(day=1)


def daterange(start: date, end: date, step_days=1):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=step_days)
