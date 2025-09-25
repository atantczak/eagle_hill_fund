# AWS S3

This code is responsible for connecting local environments to the PayFusion AWS environment and, specifically, the S3 buckets therein. This code provides read and write access if, and only if, your AWS access credentials are properly installed. Please see below for installation instructions.

<hr style="border:2px solid gray">

## <ins> Setting up S3 Connectivity </ins>

#### 1. Contact Andrew Antczak (andrew@payfusionsolutions.com) for AWS Access Keys.

#### 2. Within *this* directory, add a file titled `.env`

#### 3. Within *that* file, write the code seen below:
```python
AWS_ACCESS_KEY_ID="<INSERT AWS ACCESS KEY HERE>"
AWS_SECRET_KEY_ID="<INSERT AWS SECRET KEY HERE>"
```
If your access key was: "access_1234" <br> and your secret key was: "secret_5678" <br> then your `.env` file would be formatted precisely as follows.

```python
AWS_ACCESS_KEY_ID="access_1234"
AWS_SECRET_KEY_ID="secret_5678"
```

#### 4. Simply as a precaution: This file should never be included in a git commit.