"""Explicit licensed backend; device processing lives in XRT_devices."""
from geo_kin_core.session import resolve_session

def make_session(**config):
    return resolve_session(robot="openarm", backend="licensed", **config)
