"""Tests that spend real money.

Everything here talks to a real provider and (optionally) a real JEV endpoint.
It is excluded from the default run by the ``integration`` marker and additionally
requires ``CASCADE_RUN_INTEGRATION=1``: having keys in the environment is not
consent to bill an account, so the opt-in is explicit and separate.

Run it with:

    CASCADE_RUN_INTEGRATION=1 python -m pytest -m integration tests/integration
"""
