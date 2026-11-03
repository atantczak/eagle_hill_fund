"""eaglehillfund.com — a small FastAPI site over the reports bucket.

What it is: the fund's library of write-ups (``library/<slug>/``) and the daily
market reports (``daily-report/YYYY/MM/DD/``), both read straight from S3, plus
a public landing page. Login is enforced *in front of* the app by the load
balancer (Cognito); the app only reads the identity header the ALB forwards.

Deliberately independent of ``ehf.server.research``: the container installs only
``infra/aws/site/requirements.txt`` and copies only this package.
"""
