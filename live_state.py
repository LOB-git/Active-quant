"""Shared live-update setting, so every panel's cached data refreshes at the speed chosen in the sidebar.

dashboard.live_controls() writes LIVE on each page run. Cached loaders wrapped with live_cached() take an extra
fresh_slot argument that changes every LIVE['seconds'], so their data is never older than one live update.
"""
import time

LIVE = {'on': True, 'seconds': 10}


def fresh_slot(now=None):
    """Changes every live interval; constant while live updates are off (the cache's own ttl applies)."""
    if not LIVE['on']:
        return 0
    return int((time.time() if now is None else now) // max(1, int(LIVE['seconds'])))


def live_cached(cached):
    """Wrap an st.cache_data function whose last parameter is fresh_slot; callers keep the old arguments."""
    def call(*args, **kwargs):
        return cached(*args, fresh_slot=fresh_slot(), **kwargs)

    call.clear = lambda *args, **kwargs: cached.clear()
    call.__name__, call.__doc__, call.__wrapped__ = cached.__name__, cached.__doc__, cached
    return call
