# Dataset Directory

This directory contains event log datasets used for policy conformance checking experiments.

## BPIC 2017 Dataset

**BPI Challenge 2017** - Loan application process dataset from a Dutch financial institute.

### Download

Download the dataset from 4TU.ResearchData:
- **Direct link**: https://data.4tu.nl/ndownloader/items/34c3f44b-3101-4ea9-8281-e38905c68b8d/versions/1
- **Dataset page**: https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b

### Dataset Contents

After extraction, you should have:
- `BPI Challenge 2017.xes` - Main event log file (XES format)
- Additional application/offer files (optional)

### Usage

```bash
# Extract the downloaded zip file to this directory
unzip "BPI Challenge 2017_1_all.zip"

# Use with policy engine
cd ../policy_engine
python policy_engine.py \
  --events ../data/"BPI Challenge 2017.xes" \
  --config config/config.yaml \
  --out bpic2017_policy_log.csv
```

### Dataset Statistics

- **Cases**: ~31,509 loan applications
- **Events**: ~1,202,267 activities
- **Time span**: February 2016 - February 2017
- **Activities**: Application submission, validation, offers, acceptances
- **Resources**: Multiple employees with different roles
- **Attributes**: Case amount, activity, timestamp, resource, etc.

## .gitignore

Large dataset files are excluded from version control via `.gitignore`.
Users should download the dataset independently using the links above.
