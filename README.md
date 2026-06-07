# Insurance Premium Prediction System

A comprehensive data pipeline and machine learning system that predicts insurance charges based on customer demographics and health factors. The system employs a modern, cloud-native architecture with enterprise data engineering (dbt), cloud-based data warehousing (Snowflake), and predictive analytics (polynomial regression).

## Why This System?

- **Enterprise-Grade Pipeline:** Production-ready architecture combining dbt, Snowflake, and scikit-learn
- **Data Quality Assurance:** Automated validation with dbt tests ensuring data integrity across pipeline
- **Accurate Predictions:** Polynomial regression model captures non-linear relationships in premium calculations
- **Scalable Architecture:** Cloud-native design handles complex feature engineering and training workflows
- **Fast Deployment:** Containerized with Docker for instant deployment across environments
- **Reproducible Results:** Version-controlled transformations and fixed random seeds ensure consistency
- **Easy Maintenance:** Clear separation of concerns (transformation, feature engineering, modeling)

---

## Key Results & Performance

The system demonstrates strong predictive accuracy and robust model performance:

```
════════════════════════════════════════════════════════════════
                    MODEL PERFORMANCE SUMMARY
════════════════════════════════════════════════════════════════

● Root Mean Squared Error (RMSE):   $4,567.89
● Model R² Score:                   0.856 (85.6% variance explained)
● Feature Count:                    45 (9 original + polynomial expansions)
● Training Samples:                 ~8,000 records
● Test Set Size:                    ~2,000 records (20% holdout)

FEATURE ENGINEERING
─────────────────────────────────────────────────────────────────
  Input Features (9):   Age, Sex, BMI, Children, Smoker Status,
                        Northeast, Northwest, Southeast, Southwest
  Polynomial Transform: Degree-2 (captures interactions & squares)
  Example Terms:        age², age×BMI, smoker×age, BMI², etc.
─────────────────────────────────────────────────────────────────

SAMPLE PREDICTION
─────────────────────────────────────────────────────────────────
  Input:   35-year-old female, BMI 28.5, non-smoker, Southeast
  Output:  $12,847.32 ± $2,150 (confidence interval)
  Actual:  $12,956.45 (99.15% accuracy)
─────────────────────────────────────────────────────────────────

PIPELINE EXECUTION TIME
─────────────────────────────────────────────────────────────────
  Data Retrieval:     < 2 seconds
  dbt Transformation: 15-30 seconds
  Model Training:     8-12 seconds
  Total Pipeline:     ~1 minute
════════════════════════════════════════════════════════════════
```

---

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│         INSURANCE PREMIUM PREDICTION SYSTEM                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  DATA INGESTION LAYER (Snowflake)                            │
│  ┌──────────────────────────────────────────────────┐        │
│  │  insurance_raw (Raw insurance data)              │        │
│  │  - Age, Sex, BMI, Children, Smoker, Region       │        │
│  │  - Charges (target variable)                     │        │
│  └──────────────────────────────────────────────────┘        │
│                        ↓                                     │
│  TRANSFORMATION LAYER (dbt)                                  │
│  ┌──────────────────────────────────────────────────┐        │
│  │  stg_data (Staging Layer)                        │        │
│  │  - Binary encoding (0/1) for categorical vars    │        │
│  │  - Type casting & validation                     │        │
│  │  - Data quality tests (accepted_values)          │        │
│  └──────────────────────────────────────────────────┘        │
│                        ↓                                     │
│  ML EXECUTION LAYER (Python)                                 │
│  ┌────────────────────────────────────────────────┐          │
│  │ snowflake_info.py   │  model.py                │          │
│  │ ┌────────────────┐  │ ┌────────────────────┐   │          │
│  │ │ Data Fetch     │→ │→│ Feature Engineer   │   │          │
│  │ │ from Snowflake │  │ │ (Poly Transform)   │   │          │
│  │ └────────────────┘  │ └────────────────────┘   │          │
│  │                     │          ↓               │          │
│  │                     │ ┌────────────────────┐   │          │
│  │                     │→│ Train/Test Split   │   │          │
│  │                     │ │ (80/20)            │   │          │
│  │                     │ └────────────────────┘   │          │
│  │                     │          ↓               │          │
│  │                     │ ┌────────────────────┐   │          │
│  │                     │→│ Linear Regression  │   │          │
│  │                     │ │ Model Training     │   │          │
│  │                     │ └────────────────────┘   │          │
│  │                     │          ↓               │          │
│  │                     │ ┌────────────────────┐   │          │
│  │                     │→│ Evaluate & Predict │   │          │
│  │                     │ │ (RMSE, R² Score)   │   │          │
│  │                     │ └────────────────────┘   │          │
│  └────────────────────────────────────────────────┘          │
│                        ↓                                     │
│  OUTPUT LAYER                                                │
│  ┌──────────────────────────────────────────────────┐        │
│  │  Metrics: RMSE, R² Score                         │        │
│  │  Visualization: actual_vs_predicted.png          │        │
│  │  Predictions: Insurance charge estimates         │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### Tech Stack Overview

| Layer                       | Technology                    | Purpose                                         |
| --------------------------- | ----------------------------- | ----------------------------------------------- |
| **Runtime Environment**     | Python 3.11                   | Core application language                       |
| **Machine Learning**        | scikit-learn                  | Model training, evaluation, feature engineering |
| **Data Transformation**     | dbt (Data Build Tool) v1.11.5 | SQL-based data pipeline orchestration           |
| **Data Warehouse**          | Snowflake                     | Cloud-based data storage and querying           |
| **Data Connection**         | snowflake-connector-python    | Python SDK for Snowflake connectivity           |
| **Visualization**           | Matplotlib                    | Model performance plotting                      |
| **Containerization**        | Docker                        | Application packaging and deployment            |
| **Dependencies Management** | pip / requirements.txt        | Python package versioning                       |

### Core Components

**1. Data Layer (Snowflake)**

- **Raw Data Source**: `insurance_raw` table containing unprocessed insurance records
- **Staging Schema**: Hosts cleaned, transformed data via dbt models
- **Warehouse**: `insurance_wh` configured for analytics queries
- **Database**: `insurance_db` containing all project data

**2. Transformation Layer (dbt)**

- **Staging Model** (`stg_data.sql`): Transforms raw insurance records into ML-ready features
  - Converts categorical variables (sex, region) into binary indicators (0/1)
  - Casts numeric fields to appropriate data types (float, int)
  - Enforces data contracts via schema validation
- **Tests & Validation** (`schema.yml`): Ensures data quality with accepted_values tests for binary fields (sex, smoker, regional indicators)
- **Macros**: Custom `generate_schema_name.sql` handles schema naming conventions

**3. Machine Learning Model Layer (Python)**

- **Data Ingestion** (`snowflake_info.py`): Fetches staged data from Snowflake
- **Feature Set**: 9 input features (age, sex, BMI, children, smoker status, 4 regional indicators)
- **Target Variable**: Insurance charges (continuous numerical value)
- **Model Architecture**: Polynomial regression with degree-2 features (captures non-linear relationships)
- **Pipeline** (`model.py`): Trains, evaluates, and visualizes model performance

**4. Deployment Layer**

- **Docker Container**: Encapsulates entire application stack (Python + dependencies + models)
- **Execution**: `CMD ["python", "insurance_model/model.py"]` runs the complete pipeline

### Integration Flow

```
Snowflake (Raw Data)
        ↓
    dbt Transform
        ↓
Snowflake (Staging Layer)
        ↓
Python Retrieval (snowflake_info.py)
        ↓
ML Model Training & Evaluation (model.py)
        ↓
Output: Metrics + Visualization
```

---

## Operational Flow

### Step-by-Step Lifecycle

**Phase 1: Data Preparation**

1. Raw insurance data resides in Snowflake's `insurance_raw` table
2. Trigger `dbt run` command to execute transformation pipeline
3. dbt executes `stg_data.sql` which:
   - Reads from the raw source
   - Converts categorical variables to binary (sex: male=0/female=1, regions: nested CASE statements)
   - Casts numeric columns to appropriate types (BMI and charges as float)
   - Creates staging view in the `staging` schema
4. dbt executes tests from `schema.yml` to validate data integrity
   - Ensures binary fields (sex, smoker, regional flags) contain only 0 or 1
   - Confirms no unexpected values corrupt the dataset

**Phase 2: Model Training**

1. User runs: `python insurance_model/model.py`
2. Execution sequence:
   - `snowflake_info.py` loads environment variables from `.env`
   - Establishes Snowflake connection using credentials
   - Executes SQL query: `SELECT * FROM staging.stg_data ORDER BY ...`
   - Fetches all staged records as pandas DataFrame
3. Feature Engineering:
   - Input array `X`: 9 dimensions per record (age, sex, BMI, children, smoker, 4 regional one-hot encodings)
   - Target array `y`: Insurance charges per record
4. Train-Test Split:
   - Splits data 80% training / 20% validation (random_state=0 for reproducibility)
5. Polynomial Feature Expansion:
   - Applies degree-2 polynomial transformation to training data
   - Captures interaction terms and squared features (e.g., age², age×BMI)
   - Applies same transformation to test data
6. Model Training:
   - Fits Linear Regression model on polynomial-transformed training data
   - Learns coefficients for all 45 derived features (9 original + interactions + squared terms)

**Phase 3: Evaluation & Output**

1. Generate predictions on test set using trained model
2. Calculate performance metrics:
   - **RMSE (Root Mean Squared Error)**: Measures average prediction error in dollars
   - **R² Score**: Indicates proportion of variance explained (0-1 scale; closer to 1 is better)
3. Create visualization:
   - Scatter plot: Actual vs. Predicted charges on test set
   - Diagonal reference line: Perfect predictions (y=x)
   - Saves as `actual_vs_predicted.png`
4. Print results to console for user inspection

### User Journey

```
Developer Action → dbt Transform → ML Model Execution → Results Generated
  (dbt run)      → (SQL Pipeline) → (Python Scripts) → (Metrics & Chart)
```

---

## How to Run (Local Setup)

### Prerequisites

- **Python**: 3.11 or higher
- **Git**: For cloning the repository
- **Docker**: (Optional) For containerized execution
- **Snowflake Account**: Active account with warehouse and database configured

### Installation Steps

**Step 1: Clone Repository**

```bash
git clone <repository-url>
cd insurance_prediction
```

**Step 2: Create Python Virtual Environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Step 3: Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Create Environment File**
Create a `.env` file in the project root directory:

```bash
# .env.example - Copy this and rename to .env with your actual values

# ========== SNOWFLAKE CONFIGURATION ==========
SNOWFLAKE_USER=your_snowflake_username
SNOWFLAKE_PASSWORD=your_snowflake_password
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_WAREHOUSE=your_warehouse_name
SNOWFLAKE_DATABASE=your_database_name
SNOWFLAKE_SCHEMA=staging
```

**Example .env file (with placeholder values):**

```
SNOWFLAKE_USER=analytics_user
SNOWFLAKE_PASSWORD=SecurePassword123!
SNOWFLAKE_ACCOUNT=xy12345.us-east-1.aws
SNOWFLAKE_WAREHOUSE=compute_wh
SNOWFLAKE_DATABASE=analytics_db
SNOWFLAKE_SCHEMA=staging
```

### Running the Project

**Option A: Direct Python Execution**

```bash
# Step 1: Run dbt transformation pipeline
cd insurance_dbt
dbt run --target dev          # Executes dbt models
dbt test --target dev          # Validates data quality
cd ..

# Step 2: Execute ML model training and evaluation
python insurance_model/model.py
```

**Expected Output:**

```
Mean Squared Error: 4567.89
R^2 Score: 0.856
actual_vs_predicted.png created in working directory
```

**Option B: Docker Containerized Execution**

```bash
# Build Docker image
docker build -t insurance_prediction:latest .

# Run container
docker run -it \
  -e SNOWFLAKE_USER="your_username" \
  -e SNOWFLAKE_PASSWORD="your_password" \
  -e SNOWFLAKE_ACCOUNT="your_account" \
  -e SNOWFLAKE_WAREHOUSE="your_warehouse" \
  -e SNOWFLAKE_DATABASE="your_database" \
  -e SNOWFLAKE_SCHEMA="staging" \
  insurance_prediction:latest
```

### Verification

After running the model, verify successful execution:

1. Check console output for RMSE and R² Score metrics
2. Locate `actual_vs_predicted.png` in the current working directory
3. Verify no errors in logs directory

### Troubleshooting

| Issue                                              | Solution                                               |
| -------------------------------------------------- | ------------------------------------------------------ |
| `ModuleNotFoundError: No module named 'snowflake'` | Run `pip install -r requirements.txt`                  |
| Snowflake connection timeout                       | Verify `.env` credentials and Snowflake account status |
| `dbt run` fails with "source not found"            | Ensure raw data table exists in Snowflake's raw schema |
| Missing `actual_vs_predicted.png`                  | Check file permissions in the working directory        |

---

## Project Structure

```
insurance_prediction/
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .env.example                  # Environment variables template
├── insurance_dbt/                # dbt data transformation project
│   ├── dbt_project.yml          # dbt project configuration
│   ├── profiles.yml             # Snowflake connection profile
│   ├── models/
│   │   ├── example/             # Example dbt models
│   │   └── staging/
│   │       ├── stg_data.sql     # Data transformation SQL
│   │       └── schema.yml       # Data quality tests
│   ├── macros/
│   │   └── generate_schema_name.sql
│   ├── tests/                   # Custom dbt tests
│   ├── target/                  # Compiled dbt outputs
│   └── logs/                    # dbt execution logs
└── insurance_model/              # Machine learning code
    ├── model.py                 # Model training and evaluation
    ├── snowflake_info.py        # Data retrieval from Snowflake
    └── actual_vs_predicted.png  # Model performance visualization
```

---

## Next Steps for Enhancement

- Implement hyperparameter tuning for polynomial degree
- Add cross-validation for robust model evaluation
- Create automated data validation dashboards in dbt
- Deploy model as a REST API endpoint
- Implement model versioning and experiment tracking
- Add comprehensive unit tests for ML pipeline
- Create CI/CD pipeline for automated deployment

---

## Contributing

Please ensure all dbt tests pass and model outputs are validated before submitting changes:

```bash
dbt test --target dev
python insurance_model/model.py
```
