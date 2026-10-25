"""Offline bundle for the always-on cloud reader.

Nothing in this package talks to Cloudflare. ``export`` turns Postgres and the
retained MP3s into a directory of content-hashed JSON documents plus a
manifest; a later change publishes that directory.
"""
